import sys
import tkinter as tk
import tkinter.filedialog
from pathlib import Path

from threedattack import Scene, SceneConfig, create_scene_or_quit_wit_error
from threedattack.external_datasets import NYUV2_IM_SIZE, NYUV2_MAX_DEPTH
from threedattack.rendering_core import (
    DesiredViewpointCounts,
    ObjectTransformType,
    ThreeDPoint,
    TwoDSize,
)
from threedattack.tempfolder import GlobalTempFolder


def main() -> None:
    scene_path = get_scene_path()

    if scene_path.suffix == ".scene":
        scene = Scene.load(scene_path)
    else:
        scene = create_scene_or_quit_wit_error(
            SceneConfig(
                world_path=scene_path,
                applied_transform=None,
                depth_cap=NYUV2_MAX_DEPTH,
                n_volume_sampling_steps_along_shortest_axis=20,
                object_transform_type=ObjectTransformType.MeshBased,
                resolution=NYUV2_IM_SIZE,
                target_size_multiplier=1.01,
                viewpt_counts=DesiredViewpointCounts(
                    n_test_samples=0, n_train_samples=None, n_val_samples=None
                ),
            )
        )
    with GlobalTempFolder():
        scene.live_preview_then_quit()


def get_scene_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    else:
        filename = tkinter.filedialog.askopenfilename(
            filetypes=(("Attack results", "*.scene"), ("Simple scenes", "*.glb")),
            initialdir=".",
            title="Select a scene",
        )
        file_path = Path(filename)
        return file_path


if __name__ == "__main__":
    main()
