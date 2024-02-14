import datetime

import numpy as np

from threedattack.dataset_model import SampleType
from threedattack.external_datasets import (
    NYUV2_MAX_DEPTH,
    NyuDepthv2Dataset,
    nyu_depthv2_dataset_from_default_paths,
)
from threedattack.script_util import show_model_selector
from threedattack.target_model import (
    AsyncDepthPredictor,
    load_target_model_by_name,
    predict_aligned,
)


def main():
    target_model_name = show_model_selector()
    print(f"Waiting for the {target_model_name} predictor to start")
    zoedepth_predictor = load_target_model_by_name(target_model_name)

    print("Loading dataset")
    dataset = nyu_depthv2_dataset_from_default_paths(add_black_frame=True)

    print("Starting benchmark")
    mean_fps = _get_mean_fps_and_report_progress(dataset, zoedepth_predictor)

    print("Mean FPS: ", mean_fps)


def _get_mean_fps_and_report_progress(
    dataset: NyuDepthv2Dataset, depth_predictor: AsyncDepthPredictor
) -> float:
    elapsed_seconds: list[float] = []
    SAMPLE_TYPE = SampleType.Test

    n_sample_indexes = 21
    sample_indexes = np.random.default_rng().integers(
        0,
        dataset.get_n_samples().get_n_samples_by_type(SAMPLE_TYPE),
        size=n_sample_indexes,
    )
    sample_indexes = [int(sample_idx) for sample_idx in sample_indexes]

    for i, sample_idx in enumerate(sample_indexes):
        print(f"Processing {i+1}/{n_sample_indexes}")
        sample = dataset.get_sample(sample_idx, SAMPLE_TYPE)
        start = datetime.datetime.now()
        predict_aligned(
            depth_cap=NYUV2_MAX_DEPTH,
            images=sample.rgbds,
            predictor=depth_predictor,
        )
        end = datetime.datetime.now()
        elapsed_seconds.append((end - start).total_seconds())

    # ignore first measuremeent to
    # reduce the effect of any lazy loading
    elapsed_seconds = elapsed_seconds[1:]
    mean_elapsed_seconds = float(np.mean(elapsed_seconds))

    return 1 / mean_elapsed_seconds


if __name__ == "__main__":
    main()
