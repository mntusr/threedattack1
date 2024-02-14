import re
import sys
from typing import Callable, TypeVar

import numpy as np
import plotly.graph_objects as go

from .._attack_commons import AsyncDepthPredictor
from .._scene import Scene
from ..dataset_model import (
    DatasetLike,
    DepthsWithMasks,
    ExactSampleCounts,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
)
from ..external_datasets import NYUV2_MAX_DEPTH, NyuDepthv2Dataset
from ..rendering_core import show_depths_as_point_clouds_in_browser
from ..target_model import predict_aligned

T = TypeVar("T", bound=SamplesBase, covariant=True)


def show_interactive_depth_est_preview_and_quit_on_end(
    dataset: DatasetLike[T],
    predictor: AsyncDepthPredictor,
) -> None:
    """
    Show an interactive viewer in which the user can interactively try a model. The application exits when the user stops the viewer using Ctrl+C.

    Parameters
    ----------
    viewpt_types
        The possible types of viewpoints (or samples).
    viewpt_counts
        The number of viewpoints (or samples).
    predictor
        The target model.
    render_fn
        The function that gets the ground truth RGBD image from the viewpoint type. This function generally does not have to deal with error handling, since the viewer automatically validates the user inout.
    depth_cap
        TBD
    cam_proj_spec
        The camear projection properties.
    TODO update docs
    """

    def show_viewpt_with_idx_in_browser(
        viewpt_idx: int,
        viewpt_type: SampleType,
    ) -> None:
        rgbd = dataset.get_sample(viewpt_idx, viewpt_type).rgbds
        depth_cap = dataset.get_depth_cap()
        cam_proj_spec = dataset.get_cam_proj_spec()
        aligned_pred = predict_aligned(
            depth_cap=depth_cap, images=rgbd, predictor=predictor
        )

        show_depths_as_point_clouds_in_browser(
            cam_proj_spec=cam_proj_spec,
            depths={
                "pred": DepthsWithMasks(depths=aligned_pred, masks=rgbd.masks),
                "gt": rgbd.get_depths_with_masks(),
            },
        )

    sample_counts = dataset.get_n_samples()

    try:
        VIEWPT_TYPES_DICT = {
            "train": SampleType.Train,
            "test": SampleType.Test,
            "val": SampleType.Val,
        }
        VIEWPT_PATTERN = r"^((?:train)|(?:test)|(?:val))-(\d+)$"
        print("Number of viewpoints:")
        if sample_counts.get_n_samples_by_type(SampleType.Train) > 0:
            print("\tTraining:", sample_counts.n_train_samples)
        if sample_counts.get_n_samples_by_type(SampleType.Val) > 0:
            print("\tValidation:", sample_counts.n_val_samples)
        if sample_counts.get_n_samples_by_type(SampleType.Test) > 0:
            print("\tTesting:", sample_counts.n_test_samples)
        print("Press Ctrl+C to quit")
        print()
        while True:
            print("Type the viewpoint in train/val/test-number format: ", end="")
            input_text = input()
            pattern = re.findall(VIEWPT_PATTERN, input_text)
            if len(pattern) > 0:
                viewpt_type_name, viewpt_idx = pattern[0]
                viewpt_idx = int(viewpt_idx)

                viewpt_type_name: str = viewpt_type_name
                viewpt_type = VIEWPT_TYPES_DICT[viewpt_type_name]
                viewpt_count_for_this_type = sample_counts.get_n_samples_by_type(
                    viewpt_type
                )

                if viewpt_count_for_this_type == 0:
                    print("This viewpoint type is not enabled for this scene/dataset.")
                elif not (0 <= viewpt_idx < viewpt_count_for_this_type):
                    print(
                        f"The viewpoint is out of range. Viewpoint count for this type: {viewpt_count_for_this_type}"
                    )
                else:
                    show_viewpt_with_idx_in_browser(
                        viewpt_idx=viewpt_idx, viewpt_type=viewpt_type
                    )
    except KeyboardInterrupt:
        print("\nQuitting...")
        sys.exit(0)
