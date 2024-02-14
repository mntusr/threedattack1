from typing import Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from threedattack.dataset_model import (
    DepthsWithMasks,
    RGBsWithDepthsAndMasks,
    SamplesBase,
    SampleType,
)
from threedattack.external_datasets import (
    NyuV2Samples,
    nyu_depthv2_dataset_from_default_paths,
)
from threedattack.rendering_core import imshow, show_depths_as_point_clouds_in_browser
from threedattack.script_util import AxesArray2D, wait_for_enter
from threedattack.tempfolder import GlobalTempFolder


def main():
    with GlobalTempFolder():
        _do_main()


def _do_main():
    dataset = nyu_depthv2_dataset_from_default_paths(add_black_frame=True)
    sample = dataset.get_sample(110, SampleType.Test)
    show_depths_as_point_clouds_in_browser(
        cam_proj_spec=dataset.CAM_PROJ_SPEC,
        depths={
            "Ground truth": DepthsWithMasks(
                depths=sample.rgbds.depths, masks=sample.rgbds.masks
            )
        },
    )
    _show_sample(sample)
    wait_for_enter("Press enter to quit.")


def _show_sample(sample: NyuV2Samples):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('BTS name: "' + sample.names[0] + '"')
    axs = cast(AxesArray2D, axs)
    axs[0, 0].set_title("RGB image")
    imshow(sample.rgbds.rgbs, on=axs[0, 0], show=False)
    axs[1, 0].set_title("Depth map")
    imshow(sample.rgbds.depths, on=axs[1, 0], show=False)
    axs[0, 1].set_title("Mask")
    imshow(sample.rgbds.masks, on=axs[0, 1], show=False)
    axs[1, 1].set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()
