import argparse
from typing import Literal, Protocol, cast

from threedattack.dataset_model import SampleType
from threedattack.external_datasets import (
    NyuDepthv2Dataset,
    nyu_depthv2_dataset_from_default_paths,
)
from threedattack.losses import RawLossFn
from threedattack.script_util import (
    calculate_mean_losses_of_predictor_on_dataset,
    show_interactive_depth_est_preview_and_quit_on_end,
    show_model_selector,
)
from threedattack.target_model import (
    AsyncDepthPredictor,
    DptBeit384Predictor,
    MidasLargePredictor,
    MidasSmallPredictor,
    ZoeDepthPredictor,
    load_target_model_by_name,
)
from threedattack.tempfolder import GlobalTempFolder


def main() -> None:
    with GlobalTempFolder():
        _do_main(parse_args())


def parse_args() -> "Args":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        default=SUPPORTED_TESTING_MODES,
        choices=SUPPORTED_TESTING_MODES,
        help=f'The list of testing modes. Value "quant" means that the loss functions should be calculated on ZoeDepth on the NYUv2 dataset, then compared to the published values. Value "qual" means that a single prediction should be done on one of the images of NYUv2, then it should be shown alongside the ground truth. Default value {SUPPORTED_TESTING_MODES}',
    )
    parsed = parser.parse_args()
    return cast(Args, parsed)


class Args(Protocol):
    mode: "list[TestingMode]"


SUPPORTED_TESTING_MODES = ["qual", "quant"]
TestingMode = Literal["qual", "quant"]


def _do_main(args: Args):
    model_name = show_model_selector()
    print(f'Waiting for the "{model_name}" predictor to start')
    predictor = load_target_model_by_name(model_name)
    dataset = nyu_depthv2_dataset_from_default_paths(
        add_black_frame=model_name == ZoeDepthPredictor.MODEL_NAME
    )

    if "quant" in args.mode:
        _do_quantitative_tests_and_report(dataset, predictor)
    if "qual" in args.mode:
        _do_qualitative_tests(dataset, predictor)


def _do_quantitative_tests_and_report(
    dataset: NyuDepthv2Dataset, depth_predictor: AsyncDepthPredictor
) -> None:
    expected_loss_dict: dict[tuple[SampleType, RawLossFn], float]
    atol_dict: dict[tuple[SampleType, RawLossFn], float]
    match depth_predictor.get_name():
        case ZoeDepthPredictor.MODEL_NAME:
            expected_loss_dict = {
                (SampleType.Test, RawLossFn.RMSE): 0.277,
                (SampleType.Test, RawLossFn.D1): 0.953,
                (SampleType.Test, RawLossFn.Log10): 0.033,
            }
            atol_dict = {
                (SampleType.Test, RawLossFn.RMSE): 0.02,
                (SampleType.Test, RawLossFn.D1): 0.01,
                (SampleType.Test, RawLossFn.Log10): 0.003,
            }
        case MidasSmallPredictor.MODEL_NAME:
            expected_loss_dict = {
                (SampleType.Test, RawLossFn.D1): 0.8657,
            }
            atol_dict = {
                (SampleType.Test, RawLossFn.D1): 0.01,
            }
        case DptBeit384Predictor.MODEL_NAME:
            expected_loss_dict = {
                (SampleType.Test, RawLossFn.D1): 0.9779,
            }
            atol_dict = {
                (SampleType.Test, RawLossFn.D1): 0.01,
            }
        case MidasLargePredictor.MODEL_NAME:
            expected_loss_dict = {
                (SampleType.Test, RawLossFn.D1): 0.9129,
            }
            atol_dict = {
                (SampleType.Test, RawLossFn.D1): 0.01,
            }
        case _:
            assert False

    actual_mean_losses = calculate_mean_losses_of_predictor_on_dataset(
        dataset=dataset,
        batch_size=4,
        sample_types_and_fns=set(expected_loss_dict.keys()),
        predictor=depth_predictor,
    )

    print("Evaluation complete.")
    print()

    for sample_type_and_fn, mean_loss_val in actual_mean_losses.items():
        if sample_type_and_fn in expected_loss_dict.keys():
            _expect_mean_loss(
                sample_type_and_loss_fn=sample_type_and_fn,
                actual_loss=mean_loss_val,
                expected_loss=expected_loss_dict[sample_type_and_fn],
                atol=atol_dict[sample_type_and_fn],
            )


def _expect_mean_loss(
    sample_type_and_loss_fn: tuple[SampleType, RawLossFn],
    expected_loss: float,
    actual_loss: float,
    atol: float,
):
    pass_str = "PASS" if abs(expected_loss - actual_loss) < atol else "FAIL"
    loss_name_str = (
        sample_type_and_loss_fn[0].public_name
        + "_"
        + sample_type_and_loss_fn[1].public_name
    )
    print(
        f"mean {loss_name_str} loss test (expected={expected_loss}; actual={actual_loss}; atol={atol}) {pass_str}"
    )


def _do_qualitative_tests(
    dataset: NyuDepthv2Dataset, depth_predictor: AsyncDepthPredictor
) -> None:
    print("Starting the interactive viewer")
    show_interactive_depth_est_preview_and_quit_on_end(
        dataset=dataset, predictor=depth_predictor
    )


if __name__ == "__main__":
    main()
