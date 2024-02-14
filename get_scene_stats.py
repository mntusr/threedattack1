import sys
import tkinter as tk
from pathlib import Path

import numpy as np

from threedattack import SceneConfig, create_scene_or_quit_wit_error
from threedattack.dataset_model import SampleType
from threedattack.external_datasets import NYUV2_IM_SIZE, NYUV2_MAX_DEPTH
from threedattack.rendering_core import DesiredViewpointCounts, ObjectTransformType
from threedattack.script_util import calculate_dataset_depth_stats, show_scene_selector


def main() -> None:
    scene_path = get_scene_path()
    scene = create_scene_or_quit_wit_error(
        SceneConfig(
            world_path=scene_path,
            resolution=NYUV2_IM_SIZE,
            viewpt_counts=DesiredViewpointCounts(None, None, None),
            n_volume_sampling_steps_along_shortest_axis=1,
            object_transform_type=ObjectTransformType.MeshBased,
            target_size_multiplier=1.01,
            depth_cap=NYUV2_MAX_DEPTH,
            applied_transform=None,
        )
    )
    train_stats = calculate_dataset_depth_stats(
        dataset=scene, sample_type=SampleType.Train
    )
    val_stats = calculate_dataset_depth_stats(dataset=scene, sample_type=SampleType.Val)

    min_train_depth = np.min(train_stats.min_depths)
    max_train_depth = np.max(train_stats.max_depths)
    min_val_depth = np.min(val_stats.min_depths)
    max_val_depth = np.min(val_stats.max_depths)

    min_depth = min(min_train_depth, min_val_depth)
    max_depth = max(max_train_depth, max_val_depth)

    print("Min depth: ", min_depth)
    print("Max depth: ", max_depth)


def get_scene_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    else:
        return show_scene_selector()


if __name__ == "__main__":
    main()
