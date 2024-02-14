import sys
import tkinter as tk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from threedattack import Scene, SceneConfig, create_scene_or_quit_wit_error
from threedattack.dataset_model import SampleType
from threedattack.external_datasets import NYUV2_IM_SIZE, NYUV2_MAX_DEPTH
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    ObjectTransformType,
    PointBasedVectorFieldSpec,
    ThreeDPoint,
    TwoDSize,
    imshow,
)
from threedattack.script_util import show_scene_selector
from threedattack.tempfolder import GlobalTempFolder
from threedattack.tensor_types.npy import *


def main() -> None:
    scene_path = get_scene_path()

    rng = np.random.default_rng(356)
    for transform_type in ObjectTransformType:
        scene = create_scene_or_quit_wit_error(
            SceneConfig(
                world_path=scene_path,
                resolution=NYUV2_IM_SIZE,
                viewpt_counts=DesiredViewpointCounts(None, None, None),
                n_volume_sampling_steps_along_shortest_axis=20,
                object_transform_type=transform_type,
                target_size_multiplier=1.01,
                depth_cap=NYUV2_MAX_DEPTH,
                applied_transform=None,
            )
        )
        vector_field = _new_transform_spec(scene, rng)

        with scene.temporary_target_transform(vector_field):
            sample = scene.get_sample(0, SampleType.Train)

        fig = plt.figure()
        ax: plt.Axes = plt.gca()
        imshow(idx_im_rgbs(sample.rgbds.rgbs, n=[0]), on=ax, show=False)
        plt.title(transform_type.public_name)
        plt.show(block=True)
        plt.close(fig)
        scene.destroy_showbase()


def _new_transform_spec(
    scene: Scene, rng: np.random.Generator
) -> PointBasedVectorFieldSpec:
    n_points = 20

    rel_control_points = rng.uniform(0, 1, size=newshape_points_space(n=n_points))
    exact_control_points = (
        scene.get_target_areas()
        .get_full_area(origin_of_obj=None, include_outside_offset=False)
        .make_rel_points_absolute(rel_control_points)
    )

    exact_vectors = rng.uniform(0, 0.03, size=newshape_points_space(n=n_points))
    return PointBasedVectorFieldSpec(
        control_points=exact_control_points, vectors=exact_vectors
    )


def get_scene_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    else:
        return show_scene_selector()


if __name__ == "__main__":
    main()
