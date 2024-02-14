import argparse
import sys
import tkinter as tk
from pathlib import Path
from typing import Any, NamedTuple, cast

from threedattack.dataset_model import ExactSampleCounts
from threedattack.local_config import get_local_config_json
from threedattack.rendering_core import (
    Panda3dShowBase,
    TwoDSize,
    ViewpointSplit,
    get_viewpoints_path_for_world,
)
from threedattack.script_util import show_scene_selector


def main():
    options = get_split_generation_options()

    if options is None:
        print("No split generation options are specified.")
        return

    base = Panda3dShowBase(offscreen=True, win_size=TwoDSize(x_size=800, y_size=600))

    errors = base.load_world_from_blender(options.scene_path)

    if len(errors) > 0:
        print("Failed to load the world. Errors:")
        for error in errors:
            print(error)
        sys.exit(1)

    errors = ViewpointSplit.get_generate_errors(base, options.viewpt_counts)

    if len(errors) > 0:
        print(
            "Could not generate viewpoint split. The configuration is not valid. Errors:"
        )
        for error in errors:
            print(error)
        sys.exit(1)

    viewpts_path = get_viewpoints_path_for_world(options.scene_path)

    if viewpts_path.exists():
        print(f'The viewpoint split "{viewpts_path}" already exists.')
        sys.exit(1)

    viewpt_split = ViewpointSplit.generate(
        base=base, viewpt_counts=options.viewpt_counts, seed=None
    )
    viewpt_split.save_npz(viewpts_path)
    print("Viewpoint split generated")


class ViewpointSplitOptions(NamedTuple):
    viewpt_counts: ExactSampleCounts
    scene_path: Path


def get_split_generation_options() -> ViewpointSplitOptions | None:
    scene_path_strs = [str(scene) for scene in Path("./scenes").rglob("*.glb")] + [
        str(scene) for scene in Path("test_resources").rglob("*.glb")
    ]

    class Output:
        val: ViewpointSplitOptions | None

        def __init__(self):
            self.val = None

    output = Output()

    def is_positive_int(val) -> bool:
        if len(val) == 0:
            return False
        if str.isdigit(val):
            return int(val) >= 1
        else:
            return False

    def ok_command():
        output.val = ViewpointSplitOptions(
            viewpt_counts=ExactSampleCounts(
                n_train_samples=int(train_viewpoints_entry.get()),
                n_test_samples=int(test_viewpoints_entry.get()),
                n_val_samples=int(val_viewpoints_entry.get()),
            ),
            scene_path=Path(scene_var.get()),
        )
        window.destroy()

    window = tk.Tk()

    options_frame = tk.Frame(window)
    int_validation_command = (window.register(is_positive_int), "%P")

    tk.Label(options_frame, text="Scene").grid(row=0, column=0)
    scene_var = tk.StringVar(options_frame, value=scene_path_strs[0])
    tk.OptionMenu(options_frame, scene_var, *scene_path_strs).grid(row=0, column=1)

    tk.Label(options_frame, text="Train viewpoints").grid(row=1, column=0)
    train_viewpoints_entry = tk.Entry(
        options_frame, validate="all", validatecommand=int_validation_command
    )
    train_viewpoints_entry.insert(0, "1")
    train_viewpoints_entry.grid(row=1, column=1)

    tk.Label(options_frame, text="Test viewpoints").grid(row=2, column=0)
    test_viewpoints_entry = tk.Entry(
        options_frame,
        validate="all",
        validatecommand=int_validation_command,
    )
    test_viewpoints_entry.insert(0, "1")
    test_viewpoints_entry.grid(row=2, column=1)

    tk.Label(options_frame, text="Validation viewpoints").grid(row=3, column=0)
    val_viewpoints_entry = tk.Entry(
        options_frame, validate="all", validatecommand=int_validation_command
    )
    val_viewpoints_entry.insert(0, "1")
    val_viewpoints_entry.grid(row=3, column=1)
    options_frame.pack()

    ok_button = tk.Button(window, text="OK", command=ok_command)
    ok_button.pack()

    window.mainloop()

    return output.val


if __name__ == "__main__":
    main()
