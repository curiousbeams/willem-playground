import contextlib
import copy
import gc
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, Sequence, cast
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.io.serialize import load as autoserialize_load
from quantem.diffractive_imaging.dataset_models import DatasetModelType
from quantem.diffractive_imaging.detector_models import DetectorModelType
from quantem.diffractive_imaging.logger_ptychography import LoggerPtychography
from quantem.diffractive_imaging.object_models import ObjectModelType, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbeModelType, ProbeParametric
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher
from quantem.diffractive_imaging.ptychography_base import PtychographyBase
from quantem.diffractive_imaging.ptychography_opt import PtychographyOpt
from quantem.diffractive_imaging.ptychography_visualizations import PtychographyVisualizations
from quantem.core.visualization import show_2d


if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class Ptychography(PtychographyOpt, PtychographyVisualizations, PtychographyBase):
    """
    A class for performing phase retrieval using the Ptychography algorithm.
    """

    @classmethod
    def from_models(
        cls,
        dset: DatasetModelType,
        obj_model: ObjectModelType,
        probe_model: ProbeModelType,
        detector_model: DetectorModelType,
        logger: LoggerPtychography | None = None,
        device: str | int = "cpu",  # "gpu" | "cpu" | "cuda:X"
        verbose: int | bool = True,
        rng: np.random.Generator | int | None = None,
        wave_ground_truth: DatasetModelType | None = None, 
    ) -> Self:
        
        
        return cls(
            dset=dset,
            obj_model=obj_model,
            probe_model=probe_model,
            detector_model=detector_model,
            logger=logger,
            device=device,
            verbose=verbose,
            rng=rng,
            wave_ground_truth=wave_ground_truth,
            _token=cls._token,
        )

    @classmethod
    def from_ptychography(
        cls,
        ptycho: Self,
        obj_model: ObjectModelType | None = None,
        probe_model: ProbeModelType | None = None,
        logger: LoggerPtychography | None = None,
    ) -> Self:
        _tmp_logger = ptycho.logger
        ptycho.logger = None
        cloned = ptycho.clone()
        ptycho.logger = _tmp_logger
        if obj_model is not None:
            cloned.obj_model = obj_model
        if probe_model is not None:
            cloned.probe_model = probe_model
        if logger is not None:
            cloned.logger = logger

        cloned.reset_recon()
        return cloned

    # region --- explicit properties and setters ---

    @property
    def autograd(self) -> bool:
        return self._autograd

    @autograd.setter
    def autograd(self, autograd: bool) -> None:
        self._autograd = bool(autograd)

    # endregion --- explicit properties and setters ---

    # region --- methods ---
    # TODO reset RNG as well
    def reset_recon(self) -> None:
        super().reset_recon()
        self.obj_model.reset_optimizer()
        self.probe_model.reset_optimizer()
        self.dset.reset_optimizer()
        self.a0 = 0

    def _record_iter(self, iter_loss: float) -> None:
        self._iter_losses.append(iter_loss)
        optimizers = self.optimizers
        all_keys = set(self._iter_lrs.keys()) | set(optimizers.keys())
        for key in all_keys:
            if key in self._iter_lrs.keys():
                if key in optimizers.keys():
                    self._iter_lrs[key].append(optimizers[key].param_groups[0]["lr"])
                else:
                    self._iter_lrs[key].append(0.0)
            else:  # new optimizer
                # For new optimizers, backfill with 0.0 LR for previous iterations
                current_iter = self.num_iters - 1  # -1 because loss was just appended
                prev_lrs = [0.0] * current_iter
                prev_lrs.append(optimizers[key].param_groups[0]["lr"])
                self._iter_lrs[key] = prev_lrs

    def _reset_iter_constraints(self) -> None:
        """Reset constraint loss accumulation for all models."""
        self.obj_model.reset_iter_constraint_losses()
        self.probe_model.reset_iter_constraint_losses()
        self.dset.reset_iter_constraint_losses()

    def _soft_constraints(self) -> torch.Tensor:
        """Calculate soft constraints by calling apply_soft_constraints on each model."""
        total_loss = torch.tensor(0, device=self.device, dtype=self._dtype_real)

        obj_loss = self.obj_model.apply_soft_constraints(
            self.obj_model.obj, mask=self.obj_model.mask
        )
        total_loss += obj_loss

        probe_loss = self.probe_model.apply_soft_constraints(self.probe_model.probe)
        total_loss += probe_loss

        dataset_loss = self.dset.apply_soft_constraints(self.dset.descan_shifts)
        total_loss += dataset_loss

        return total_loss

    # endregion --- methods ---

    # region --- reconstruction ---
    
    def find_brightfield_disk(
        self,
    ):
        plt.plot()
    def reconstruct(
        self,
        num_iters: int = 0,
        reset: bool = False,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        constraints: dict = {},
        batch_size: int | None = None,
        store_snapshots: bool | None = None,
        store_snapshots_every: int | None = None,
        device: Literal["cpu", "gpu"] | None = None,
        autograd: bool = True,
        autograd_random: bool = False,
        disk_phase: bool = False,
        loss_type: Literal[
            "l2_amplitude", "l1_amplitude", "l2_intensity", "l1_intensity", "poisson"
        ] = "l2_amplitude",
        wave_ground_truth = None
    ) -> Self:
        """
        reason for having a single reconstruct() is so that updating things like constraints
        or recon_types only happens in one place, reason for having separate reoconstruction_
        methods would be to simplify the flags for this and not have to include all

        """
        # TODO maybe make an "process args" method that handles things like:
        # mode, store_iterations, device,
        self._check_preprocessed()
        if device is not None:
            self.to(device)
        self.batch_size = batch_size
        self.store_snapshot_every = store_snapshots_every
        if store_snapshots_every is not None and store_snapshots is None:
            self.store_snapshots = True
        else:
            self.store_snapshots = store_snapshots

        if reset:
            self.reset_recon()
        self.constraints = constraints

        new_scheduler = reset
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.set_optimizers()
            new_scheduler = True

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            new_scheduler = True

        if new_scheduler:
            self.set_schedulers(self.scheduler_params, num_iter=num_iters)

        self.dset._set_targets(loss_type)
        self.compute_propagator_arrays()  # required to avoid issue if stopped learning probe tilt
        batcher = SimpleBatcher(
            self.dset.num_gpts,
            self.batch_size,
            rng=self.rng,
            val_ratio=self.val_ratio,
            val_mode=self.val_mode,
        )
        pbar = tqdm(range(num_iters), disable=not self.verbose)
        self.dset.learn_descan = False

        for a0 in pbar:
            consistency_loss = 0.0
            total_loss = 0.0
            self._reset_iter_constraints()
            self.a0 = a0
            for batch_indices in batcher:
                self.zero_grad_all()
                patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                    self.dset.forward(batch_indices, self.obj_padding_px)
                )
                shifted_probes = self.probe_model.forward(positions_px_fractional)
                obj_patches = self.obj_model.forward(patch_indices)
                propagated_probes, overlap = self.forward_operator(
                    obj_patches, shifted_probes, descan_shifts
                )
                pred_intensities = self.detector_model.forward(overlap)         # pred_intensities: fourier space, centered, no phase, [batch, h, w]
                batch_consistency_loss, targets = self.error_estimate(           # targets: fourier space, centered, no phase, [batch, h, w]
                    pred_intensities,
                    batch_indices,
                    loss_type=loss_type,
                )

                batch_soft_constraint_loss = self._soft_constraints()
                batch_loss = batch_consistency_loss + batch_soft_constraint_loss

                self.backward(
                    batch_loss,
                    autograd,
                    obj_patches,
                    propagated_probes,
                    overlap,
                    patch_indices,
                    targets,
                    autograd_random,
                    disk_phase,
                    batch_indices
                )
                self.step_optimizers()
                consistency_loss += batch_consistency_loss.item()
                total_loss += batch_loss.item()

            num_batches = len(batcher)
            total_loss = total_loss / num_batches
            consistency_loss = consistency_loss / num_batches

            # Validation pass (no gradient, no optimizer steps)
            val_loss = None
            if batcher.has_validation:
                val_consistency_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_indices in batcher.iter_val():
                        patch_indices, _positions_px, positions_px_fractional, descan_shifts = (
                            self.dset.forward(batch_indices, self.obj_padding_px)
                        )
                        shifted_probes = self.probe_model.forward(positions_px_fractional)
                        obj_patches = self.obj_model.forward(patch_indices)
                        _propagated_probes, overlap = self.forward_operator(
                            obj_patches, shifted_probes, descan_shifts
                        )
                        pred_intensities = self.detector_model.forward(overlap)
                        batch_val_loss, _ = self.error_estimate(
                            pred_intensities, batch_indices, loss_type=loss_type
                        )
                        val_consistency_loss += batch_val_loss.item()
                        val_batches += 1
                if val_batches > 0:
                    val_loss = val_consistency_loss / val_batches
                    self._iter_val_losses.append(val_loss)

            self._record_iter(total_loss)  # TODO record val loss as well

            # Step schedulers with current loss
            self.step_schedulers(total_loss)

            if self.store_snapshots and (a0 % self.store_snapshot_every) == 0:
                self._store_current_iter_snapshot()

            if self.logger is not None:
                self.logger.log_iter(
                    self.obj_model,
                    self.probe_model,
                    self.dset,
                    self.num_iters - 1,
                    consistency_loss,
                    num_batches,
                    self._get_current_lrs(),
                )

            if val_loss is not None:
                pbar.set_description(
                    f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}, Val: {val_loss:.3e}"
                )
            else:
                pbar.set_description(f"Iter {a0 + 1}/{num_iters}, Loss: {total_loss:.3e}")

        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    def _get_current_lrs(self) -> dict[str, float]:
        return {
            param_name: optimizer.param_groups[0]["lr"]
            for param_name, optimizer in self.optimizers.items()
            if optimizer is not None
        }

    def backward(
        self,
        loss: torch.Tensor,
        autograd: bool,
        obj_patches: torch.Tensor,
        propagated_probes: torch.Tensor,
        overlap: torch.Tensor,
        patch_indices: torch.Tensor,
        amplitudes: torch.Tensor,
        autograd_random: bool,
        disk_phase,
        batch_indices
    ):
        if autograd:
            loss.backward()
            # scaling pixelated ad gradients to closer match analytic
            if isinstance(self.obj_model, ObjectPixelated):
                obj_grad_scale = self.dset.upsample_factor**2 / 2  # factor of 2 from l2 grad
                self.obj_model._obj.grad.mul_(obj_grad_scale)  # type:ignore

            if isinstance(self.probe_model, ProbeParametric):
                probe_grad_scale = np.sqrt(self.probe_model._mean_diffraction_intensity)
                for par in self.probe_model.params:
                    par.grad.mul_(probe_grad_scale)  # type:ignore

        else:
            gradient = self.gradient_step(amplitudes, overlap, autograd_random, disk_phase, batch_indices)
            prop_gradient = self.obj_model.backward(
                gradient,
                obj_patches,
                propagated_probes,
                self._propagators,
                patch_indices,
            )
            self.probe_model.backward(prop_gradient, obj_patches)

    def gradient_step(self, amplitudes, overlap, autograd_random, disk_phase, batch_indices):
        """Computes analytical gradient using the Fourier projection modified overlap"""
        modified_overlap, masked_overlap = self.fourier_projection(amplitudes, overlap, autograd_random, disk_phase, batch_indices)
        return modified_overlap - masked_overlap

    def fourier_projection(self, measured_amplitudes, overlap_array, autograd_random, disk_phase, batch_indices):
        """Replaces the Fourier amplitude of overlap with the measured data."""
        # corner centering measured amplitudes
        measured_amplitudes = torch.fft.fftshift(measured_amplitudes, dim=(-2, -1))     # former targets: fourier space, centered->left corner, no phase, [batch, h, w] 
        fourier_overlap = torch.fft.fft2(overlap_array, norm="ortho")                   # overlap: real space->fourier space, left corner, has phase, [nprobes, batch_size, h, w]

        if self.num_probes == 1:  # faster
            # in __init__ or reset_recon:
            if type(autograd_random) == float or autograd_random == 1.0+0.0j:
                fourier_modified_overlap = measured_amplitudes * torch.ones_like(fourier_overlap, device=self.device)
                masked_overlap_array = overlap_array

            elif autograd_random == 'mean':
                fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap).mean(dim=-3)  # shape: [1, 192, 192]
                )
                masked_overlap_array = overlap_array

            elif autograd_random == 'random':
                fourier_modified_overlap = measured_amplitudes * torch.exp(
                    1.0j * torch.rand_like(fourier_overlap, dtype=torch.float) *0.01 -(0.005)
                )
                masked_overlap_array = overlap_array

            elif autograd_random == 'disk_distinction':
                fourier_modified_overlap, masked_overlap_array =  self.angle_per_disk(fourier_overlap, disk_phase, overlap_array, batch_indices)
                fourier_modified_overlap = fourier_modified_overlap * measured_amplitudes

            else:
                masked_overlap_array = overlap_array
                fourier_modified_overlap = measured_amplitudes * torch.exp(
                1.0j * torch.angle(fourier_overlap)
                )


        else:  # necessary for mixed state # TODO check this with normalization
            farfield_amplitudes = self.estimate_amplitudes(overlap_array, corner_centered=True)
            farfield_amplitudes[farfield_amplitudes == 0] = torch.inf
            amplitude_modification = measured_amplitudes / farfield_amplitudes
            fourier_modified_overlap = amplitude_modification[None] * fourier_overlap
            masked_overlap_array = overlap_array
        return torch.fft.ifft2(fourier_modified_overlap, norm="ortho"), masked_overlap_array

    def angle_per_disk(self, fourier_overlap, disk_phase, overlap_array, batch_indices):

        masked = torch.zeros_like(fourier_overlap, device=self.device)
        masked_overlap_array = torch.zeros_like(overlap_array, device=self.device)
        fourier_modified_overlap = torch.zeros_like(overlap_array, device=self.device)
        fourier_modified_overlap_masked = torch.zeros_like(overlap_array, device=self.device)
        

        combined_mask = self.dset.disk_masks[batch_indices]
        
        if disk_phase == 'disk_phase':

            combined_mask = torch.fft.ifftshift(combined_mask)              
            masked[0, batch_indices] = torch.where(combined_mask, fourier_overlap[0, batch_indices], torch.zeros_like(fourier_overlap[0, batch_indices]))

            masked_overlap_array[0, batch_indices] = torch.where(combined_mask, fourier_overlap[0, batch_indices], torch.zeros_like(overlap_array[0, batch_indices]))
            fourier_modified_overlap[0, batch_indices] = torch.where(combined_mask, torch.exp(
                1.0j * torch.angle(fourier_overlap[0,batch_indices])), torch.zeros_like(overlap_array[0, batch_indices])
            )
            print(self.dset.amplitudes.shape)
            show_2d(combined_mask[100])
            plt.show()
            show_2d(masked[0, 100])
            plt.show()
            show_2d(fourier_modified_overlap[0, 100])
            plt.show()
            show_2d(fourier_overlap[0, 100])
            plt.show()
            fourier_modified_overlap_masked[0,batch_indices] = torch.where(combined_mask, fourier_modified_overlap[0, batch_indices], torch.zeros_like(fourier_modified_overlap[0, batch_indices]))

        elif disk_phase == 'mean':
            phase = torch.angle(fourier_overlap[0, batch_idx])
            output = torch.zeros(H, W, device=self.device)

            for (y, x) in disk_pos:
                dist_sq = (xs - x) ** 2 + (ys - y) ** 2
                disk_mask = dist_sq <= disk_radius ** 2
                mean_phase = phase[disk_mask].mean().item()  # force scalar
                output = torch.where(disk_mask, mean_phase, torch.zeros(H, W, device=self.device))
                combined_mask |= disk_mask
            combined_mask = torch.fft.ifftshift(combined_mask)
            masked[0, batch_idx] = torch.where(combined_mask, fourier_overlap[0, batch_idx].abs() * torch.exp(1.0j * output), torch.zeros_like(fourier_overlap[0, batch_idx]))



            masked_overlap_array[0, batch_idx] = torch.where(combined_mask, fourier_overlap[0, batch_idx], torch.zeros_like(overlap_array[0, batch_idx]))

            fourier_modified_overlap[0, batch_idx] = torch.exp(
                1.0j * torch.angle(masked[0,batch_idx])
            ) 
            fourier_modified_overlap_masked[0,batch_idx] = torch.where(combined_mask, fourier_modified_overlap[0, batch_idx], torch.zeros_like(fourier_modified_overlap[0, batch_idx]))



        elif disk_phase == 'intensity':
            for (x, y) in disk_pos:
                dist_sq = (xs - x) ** 2 + (ys - y) ** 2
                combined_mask |= (dist_sq <= disk_radius ** 2)

            # Set pixels inside any disk to 1, outside stays 0
            masked[0, batch_idx] = combined_mask.to(fourier_overlap.dtype)

        else:
            for (x, y) in disk_pos:
                dist_sq = (xs - x) ** 2 + (ys - y) ** 2
                combined_mask |= (dist_sq < disk_radius ** 2)

            masked[0, batch_idx] = fourier_overlap[0, batch_idx] * combined_mask

        return fourier_modified_overlap_masked, torch.fft.ifft2(masked_overlap_array)
    # endregion --- reconstruction ---

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: str | type | Sequence[str | type] = (),
        compression_level: int | None = 4,
        save_raw_data: bool = False,
        verbose: int | bool = True,
    ):
        """
        Save the ptychography object, optionally excluding raw dataset data.

        By default, this method saves the ptychography object without the raw dataset
        to save space and allow for dataset reloading. Use save_raw_data=True if you
        want to include the complete dataset.

        When saving without raw data, the system automatically saves:
        - Dataset file path and file type
        - All preprocessing parameters (CoM fitting, rotation, padding, etc.)
        - Reconstruction state (losses, constraints, etc.)

        On load, if no dataset is provided, the system will automatically:
        - Reload the dataset from the saved file path
        - Reapply all preprocessing with the exact same parameters
        - Restore the reconstruction state

        Parameters
        ----------
        path : str | Path
            Path to save the object
        mode : Literal["w", "o"]
            Write mode ('w' for write, 'o' for overwrite)
        store : Literal["auto", "zip", "dir"]
            Storage format
        skip : str | type | Sequence[str | type]
            Additional items to skip during serialization
        compression_level : int | None
            Compression level for zip storage
        save_raw_data : bool
            Whether to save the raw dataset data (default: False)

        Examples
        --------
        # Save without raw data (default behavior) - includes dataset metadata
        ptycho.save("my_reconstruction.zip")

        # Save with raw data included
        ptycho.save("my_reconstruction_with_data.zip", save_raw_data=True)

        # Load a saved reconstruction - automatically reloads dataset
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip")

        # Load and move to GPU
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", device="gpu")

        # Load with custom dataset (overrides automatic reloading)
        loaded_ptycho = Ptychography.from_file("my_reconstruction.zip", dset=my_dataset)

        """
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip)

        # Always skip raw dataset data unless explicitly requested
        if not save_raw_data:
            skip.extend(
                [
                    "_dset",  # Skip the dataset object itself
                    "dset",  # Skip dataset references
                ]
            )

            # Save dataset metadata for automatic reloading
            self._dataset_metadata = {
                "file_path": str(self.dset.dset.file_path) if self.dset.dset.file_path else None,
                "preprocessing_params": self.dset._preprocessing_params,
                "learned_scan_positions_px": self.dset.scan_positions_px.data.cpu(),
                "learned_descan_shifts": self.dset.descan_shifts.data.cpu(),
            }

        # Add other common skips for ptychography objects
        skips = skip

        current_device = self.device
        self.to("cpu")

        if self.verbose and verbose:
            print(f"Saving ptychography object to {Path(path).resolve()}")

        super().save(
            path,
            mode=mode,
            store=store,
            skip=skips,
            compression_level=compression_level,
        )

        self.to(current_device)  # TODO figure out why this isn't working for DDIP sometimes?

        # Clean up temporary metadata
        if not save_raw_data and hasattr(self, "_dataset_metadata"):
            delattr(self, "_dataset_metadata")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dset: DatasetModelType | None = None,
        device: str | int | None = None,
        verbose: int | bool | None = None,
        auto_reload_dataset: bool = True,
    ):
        """
        Load a ptychography object from a saved file.

        Parameters
        ----------
        path : str | Path
            Path to the saved ptychography object
        dset : DatasetModelType | None
            Dataset to use (if None and auto_reload_dataset=True, will try to reload from saved metadata)
        device : str | int | None
            Device to load the object on
        verbose : int | bool | None
            Verbosity level
        rng : np.random.Generator | int | None
            Random number generator
        auto_reload_dataset : bool
            Whether to automatically reload and preprocess the dataset from saved metadata

        Returns
        -------
        Ptychography
            Loaded ptychography object
        """
        # Load the base object without the dataset
        ptycho = cls._recursive_load_from_path(path)

        if not isinstance(ptycho, Ptychography):
            raise ValueError("Loaded object is not a Ptychography object")

        # If no dataset was provided, try to reload it from saved metadata
        if dset is None and auto_reload_dataset and not hasattr(ptycho, "dset"):
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                file_path = metadata.get("file_path")

                if file_path:
                    # Import here to avoid circular imports
                    from quantem.core.io.file_readers import read_4dstem
                    from quantem.diffractive_imaging.dataset_models import (
                        PtychographyDatasetRaster,
                    )

                    # Reload the dataset
                    print(f"reloading dataset from {file_path}", end="\r")
                    try:
                        raw_dset = read_4dstem(file_path)
                    except (ValueError, ModuleNotFoundError) as _e:
                        try:
                            raw_dset = autoserialize_load(file_path)
                            raw_dset.file_path = file_path  # legacy support
                        except Exception as e:
                            raise ValueError(
                                f"Could not automatically reload dataset from {file_path}: {e}"
                            )

                    dset = PtychographyDatasetRaster.from_dataset4dstem(
                        raw_dset, verbose=verbose or 1
                    )
                    # Apply preprocessing with saved parameters
                    preprocessing_params = metadata.get("preprocessing_params", {})
                    _v = dset.verbose
                    dset.verbose = 0
                    dset.preprocess(**preprocessing_params)
                    dset.verbose = _v

                    print(f"Successfully reloaded dataset from {file_path}")
                else:
                    dset = None
            else:
                print("Warning: No dataset metadata found in saved object.")
                dset = None
        elif dset is not None:
            dset._set_initial_scan_positions_px(ptycho.obj_padding_px)
            dset._set_patch_indices(ptycho.obj_padding_px)
            if hasattr(ptycho, "_dataset_metadata") and ptycho._dataset_metadata:
                metadata = ptycho._dataset_metadata
                # preserve learned scan positions and descan shifts
                if "learned_scan_positions_px" in metadata:
                    dset.scan_positions_px.data = metadata["learned_scan_positions_px"]
                if "learned_descan_shifts" in metadata:
                    dset.descan_shifts.data = metadata["learned_descan_shifts"]

        # check if dset was attached to ptycho object
        if dset is not None:
            ptycho.dset = dset
        elif not (hasattr(ptycho, "_dset") and ptycho._dset is not None):
            warn(
                "No dataset provided and could not automatically reload dataset.\n"
                "Please provide a dataset parameter or ensure the object was saved with dataset metadata.\n"
                "Many functionalities will not work without the dataset attached."
            )
            # raise ValueError(
            #     "No dataset provided and could not automatically reload dataset. "
            #     "Please provide a dataset parameter or ensure the object was saved with dataset metadata."
            # )

        if device is not None:
            ptycho.to(device)

        return ptycho

    @classmethod
    def _recursive_load_from_path(cls, path: str | Path):
        """Helper method to load an object from a path using AutoSerialize."""
        return autoserialize_load(path)

    def clone(self, device: str | int = "cpu") -> Self:  # TODO make this faster
        """
        Create a deep-copy clone of this Ptychography instance.

        The clone is placed on CPU by default (device="cpu"). You can override
        the output device by passing a different device string.

        This method first attempts a Python deepcopy for speed. If that fails
        (e.g., due to non-copyable objects), it falls back to serializing the
        object to a temporary file and reloading it, which is robust and includes
        the dataset by default.
        """
        try:
            cloned: Self = copy.deepcopy(self)
        except Exception:
            # Robust fallback: save then reload including raw dataset data so that
            # the in-memory state is fully preserved without relying on external files.
            tmp_path = (
                Path(tempfile.gettempdir()) / f"ptycho_clone_{self.rng.integers(int(1e7))}.zip"
            )
            try:
                self.save(tmp_path, mode="o", store="zip", save_raw_data=True, verbose=0)
                cloned = cast(
                    Self, Ptychography.from_file(tmp_path, device=None, auto_reload_dataset=False)
                )
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()

        if self.logger is not None:
            cloned.logger = self.logger.clone()

        cloned.to(device)
        return cloned
