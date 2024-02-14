from pathlib import Path

import numpy as np

from threedattack import Scene, SceneConfig, create_scene_or_quit_wit_error
from threedattack.dataset_model import SampleType
from threedattack.external_datasets import NYUV2_MAX_DEPTH
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    ObjectTransformType,
    TwoDSize,
    get_twod_area_masks,
    imshow,
)


def main() -> None:
    # SCENE_PATH = Path("test_resources/test_scene.glb")
    resolution = TwoDSize(800, 600)
    SCENE_PATH = Path("scenes/room1.glb")
    scene = create_scene_or_quit_wit_error(
        SceneConfig(
            world_path=SCENE_PATH,
            resolution=resolution,
            viewpt_counts=DesiredViewpointCounts(None, None, None),
            n_volume_sampling_steps_along_shortest_axis=1,
            object_transform_type=ObjectTransformType.MeshBased,
            target_size_multiplier=1.01,
            depth_cap=NYUV2_MAX_DEPTH,
            applied_transform=None,
        )
    )

    sample = scene.get_sample(200, SampleType.Train)
    target_obj_mask = get_twod_area_masks(sample.target_obj_areas_on_screen, resolution)
    rgbs_masked = sample.rgbds.rgbs * target_obj_mask

    imshow(rgbs_masked)


if __name__ == "__main__":
    main()
