from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, cast

import h5py
import numpy as np
import scipy

from .._typing import type_instance
from ..dataset_model import (
    CamProjSpec,
    DatasetLike,
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
    SampleTypeError,
)
from ..local_config import get_local_config_json
from ..rendering_core import (
    ThreeDCoordSysConvention,
    TwoDSize,
    get_matrix_to_coord_system_from_standard,
)
from ..tensor_types.idx import *

NYUV2_IM_SIZE = TwoDSize(x_size=640, y_size=480)


def _get_nyuv2_proj_spec() -> CamProjSpec:
    """
    Get the intrinsic matrix for NYUv2.

    Returns
    -------
    v
        The projection properties.
    """
    nyuv2_original_intrinsic_mat = np.array(
        [
            [525.0, 0.0, 319.5, 0],
            [0.0, 525.0, 239.5, 0],
            [0.0, 0.0, 1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    corrd_sys_transform_from_zup = get_matrix_to_coord_system_from_standard(
        ThreeDCoordSysConvention.YupLeftHanded
    )

    drop_extra_dim = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )

    total_proj_mat = (
        drop_extra_dim @ nyuv2_original_intrinsic_mat @ corrd_sys_transform_from_zup
    )

    return CamProjSpec(
        proj_mat=total_proj_mat,
        im_left_x_val=0,
        im_right_x_val=NYUV2_IM_SIZE.x_size,
        im_bottom_y_val=0,
        im_top_y_val=NYUV2_IM_SIZE.y_size,
    )


def nyu_depthv2_dataset_from_default_paths(
    add_black_frame: bool,
) -> "NyuDepthv2Dataset":
    paths_json = get_local_config_json()
    return NyuDepthv2Dataset(
        labeled_mat_path=Path(paths_json.nyuv2_labeled_mat),
        splits_mat_path=Path(paths_json.nyuv2_splits_mat),
        add_black_frame=add_black_frame,
    )


NYUV2_MIN_DEPTH = 0.001
NYUV2_MAX_DEPTH = 10


class NyuDepthv2Dataset:
    MIN_DEPTH_EVAL = 0.001
    CAM_PROJ_SPEC = _get_nyuv2_proj_spec()

    def __init__(
        self, labeled_mat_path: Path, splits_mat_path: Path, add_black_frame: bool
    ):
        h5_file = h5py.File(str(labeled_mat_path), "r")
        train_test = scipy.io.loadmat(str(splits_mat_path))

        self.test_im_idxs = [int(x) for x in train_test["testNdxs"]]
        self.raw_depths = cast(h5py.Dataset, h5_file["rawDepths"])
        self.images = cast(h5py.Dataset, h5_file["images"])
        self.scenes: list[str] = ["".join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file["sceneTypes"][0]]  # type: ignore
        self.add_black_frame = add_black_frame

    def get_sample(self, idx: int, sample_type: SampleType) -> "NyuV2Samples":
        if sample_type != SampleType.Test:
            raise SampleTypeError("Only test samples are supported by this dataset.")

        test_im_idx = self.test_im_idxs[idx] - 1
        scene_name = f"test/{self.scenes[test_im_idx]}/rgb_{test_im_idx:05d}.jpg"

        rgbs_without_black_boundary: np.ndarray = self.images[test_im_idx].transpose(
            [0, 2, 1]
        )
        if self.add_black_frame:
            rgbs = np.zeros_like(rgbs_without_black_boundary, dtype=np.uint8)
            rgbs[:, 7:474, 7:632] = rgbs_without_black_boundary[:, 7:474, 7:632]
        else:
            rgbs = rgbs_without_black_boundary
        rgbs = np.expand_dims(rgbs, 0)
        rgbs = rgbs.astype(np.float32) / 255

        gt_depth = self.raw_depths[test_im_idx]
        gt_depth = np.expand_dims(gt_depth, axis=(0, 1))
        gt_depth = gt_depth.transpose([0, 1, 3, 2])

        valid_masks = np.logical_and(
            gt_depth > self.MIN_DEPTH_EVAL, gt_depth < NYUV2_MAX_DEPTH
        )
        eval_masks = np.zeros(shape=valid_masks.shape)
        eval_masks[0, 0, 45:471, 41:601] = 1

        total_masks = np.logical_and(valid_masks, eval_masks)

        return NyuV2Samples(
            names=[scene_name],
            rgbds=RGBsWithDepthsAndMasks(rgbs=rgbs, depths=gt_depth, masks=total_masks),
        )

    def get_n_samples(self) -> ExactSampleCounts:
        return ExactSampleCounts(
            n_train_samples=0,
            n_val_samples=0,
            n_test_samples=self._get_n_test_samples(),
        )

    def get_cam_proj_spec(self) -> CamProjSpec:
        return self.CAM_PROJ_SPEC

    def get_depth_cap(self) -> float:
        return NYUV2_MAX_DEPTH

    def get_samples(
        self, idxs: Sequence[int] | slice, sample_type: "SampleType"
    ) -> "NyuV2Samples":
        if isinstance(idxs, slice):
            idx_iterable = range(*idxs.indices(self._get_n_test_samples()))
        else:
            idx_iterable = idxs

        items: "list[NyuV2Samples]" = []
        for idx in idx_iterable:
            items.append(self.get_sample(idx, sample_type))

        rgbds = RGBsWithDepthsAndMasks(
            rgbs=np.concatenate([item.rgbds.rgbs for item in items], axis=DIM_IM_N),
            depths=np.concatenate([item.rgbds.depths for item in items], axis=DIM_IM_N),
            masks=np.concatenate([item.rgbds.masks for item in items], axis=DIM_IM_N),
        )
        names = [item.names[0] for item in items]
        return NyuV2Samples(names=names, rgbds=rgbds)

    def _get_n_test_samples(self) -> int:
        return len(self.test_im_idxs)


class NyuV2Samples(SamplesBase):
    def __init__(self, names: list[str], rgbds: RGBsWithDepthsAndMasks):
        super().__init__(rgbds)
        self.names = names


def expect_loss(loss_name: str, expected_loss: float, actual_loss: float, atol: float):
    pass_str = "PASS" if abs(expected_loss - actual_loss) < atol else "FAIL"
    print(
        f"{loss_name} loss test (expected={expected_loss}; actual={actual_loss}, atol={atol}) {pass_str}"
    )


if TYPE_CHECKING:
    v: DatasetLike[NyuV2Samples] = type_instance(NyuDepthv2Dataset)
