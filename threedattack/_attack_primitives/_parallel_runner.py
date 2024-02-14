from typing import Callable, Generic, TypeVar, cast

TRenderOutput = TypeVar("TRenderOutput")
TPredictionFuture = TypeVar("TPredictionFuture")
TPredictionGot = TypeVar("TPredictionGot")
TShared = TypeVar("TShared")


class ParallelRunner(
    Generic[TRenderOutput, TShared, TPredictionFuture, TPredictionGot]
):
    """
    Runs the specified functions in order and makes sure that the prediction is parallelized with everything else.

    This class assumes that only one prediction might be done at a time.

    The logical order of the functions:

    1. ``render_fn``
    2. ``prediction_fn_async``
    3. ``acquire_pred_future_fn``
    4. ``eval_sink_fn``

    These functions may be run out of order to make parallellization possible.

    The order of the calls of the functions **individually** matches to the order of the processed items.

    Parameters
    ----------
    render_fn
        This function renders a new image and gets the ground truth data based on the specified index.
    prediction_fn_async
        Starts the prediction.
    acquire_pred_future_fn
        Blocks until the prediction is done, then gets the prediction.
    eval_sink_fn
        Do the evaluation and the final processing.
    """

    def __init__(
        self,
        render_fn: Callable[[int], tuple[TRenderOutput, TShared]],
        prediction_fn_async: Callable[[TRenderOutput, TShared], TPredictionFuture],
        acquire_pred_future_fn: Callable[[TPredictionFuture, TShared], TPredictionGot],
        eval_sink_fn: Callable[[TPredictionGot, TShared], None],
    ):
        self.render_fn = render_fn
        self.prediction_fn_async = prediction_fn_async
        self.eval_sink_fn = eval_sink_fn
        self.acquire_pred_future_fn = acquire_pred_future_fn

    def run(self, n_items: int):
        pipeline_state: tuple[
            tuple[TRenderOutput, TShared] | None,
            tuple[TPredictionFuture, TShared] | None,
        ] = cast(
            tuple[
                tuple[TRenderOutput, TShared] | None,
                tuple[TPredictionFuture, TShared] | None,
            ],
            (None, None),
        )

        item_idx = 0
        while pipeline_state != (None, None) or item_idx < n_items:
            if pipeline_state[1] is not None:
                shared_data = pipeline_state[1][1]
                got_prediction = self.acquire_pred_future_fn(*pipeline_state[1])
                if pipeline_state[0] is not None:
                    pred_future_input = pipeline_state[0]
                    pred_future_next = (
                        self.prediction_fn_async(*pred_future_input),
                        pipeline_state[0][1],
                    )
                else:
                    pred_future_next = None
                self.eval_sink_fn(got_prediction, shared_data)
            else:
                if pipeline_state[0] is not None:
                    pred_future_input = pipeline_state[0]
                    pred_future_next = (
                        self.prediction_fn_async(*pred_future_input),
                        pipeline_state[0][1],
                    )
                else:
                    pred_future_next = None

            if item_idx < n_items:
                rendered_next = self.render_fn(item_idx)
                item_idx += 1
            else:
                rendered_next = None

            pipeline_state = (rendered_next, pred_future_next)
