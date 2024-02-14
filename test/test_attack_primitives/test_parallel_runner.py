import unittest

from threedattack._attack_primitives import ParallelRunner


class TestParallelRunner(unittest.TestCase):
    def test_parallel_runner(self):
        class NumKeeper:
            def __init__(self, initial_val: int):
                self.val = initial_val

        actual_results: list[tuple[int, str]] = []
        N_ITEMS = 10

        parallel_predictor_call_count_keeper = NumKeeper(0)
        render_idx_keeper = NumKeeper(-1)
        pred_adync_idx_keeper = NumKeeper(-1)
        acquire_pred_idx_keeper = NumKeeper(-1)
        sink_idx_keeper = NumKeeper(-1)
        n_prediction_parallel_with_render = NumKeeper(0)
        n_prediction_parallel_with_eval = NumKeeper(0)

        def render_fn(idx: int) -> tuple[int, str]:
            self.assertEqual(render_idx_keeper.val + 1, idx)
            render_idx_keeper.val += 1
            if parallel_predictor_call_count_keeper.val > 0:
                n_prediction_parallel_with_render.val += 1
            return idx, f"shared_{idx}"

        def prediction_fn_async(idx: int, shared: str) -> int:
            self.assertEqual(pred_adync_idx_keeper.val + 1, idx)
            pred_adync_idx_keeper.val += 1
            self.assertEqual(shared, f"shared_{idx}")
            self.assertEqual(parallel_predictor_call_count_keeper.val, 0)
            parallel_predictor_call_count_keeper.val += 1
            return idx

        def acquire_pred_future_fn(idx: int, shared: str) -> int:
            self.assertEqual(acquire_pred_idx_keeper.val + 1, idx)
            acquire_pred_idx_keeper.val += 1
            self.assertEqual(shared, f"shared_{idx}")
            self.assertEqual(parallel_predictor_call_count_keeper.val, 1)
            parallel_predictor_call_count_keeper.val -= 1
            return idx

        def eval_sink_fn(idx: int, shared: str) -> None:
            self.assertEqual(sink_idx_keeper.val + 1, idx)
            sink_idx_keeper.val += 1
            if parallel_predictor_call_count_keeper.val > 0:
                n_prediction_parallel_with_eval.val += 1
            self.assertEqual(shared, f"shared_{idx}")
            actual_results.append((idx, shared))

        parallel_runner = ParallelRunner(
            render_fn=render_fn,
            prediction_fn_async=prediction_fn_async,
            acquire_pred_future_fn=acquire_pred_future_fn,
            eval_sink_fn=eval_sink_fn,
        )

        parallel_runner.run(n_items=N_ITEMS)

        expected_results = [(idx, f"shared_{idx}") for idx in range(N_ITEMS)]

        self.assertEqual(actual_results, expected_results)
        self.assertGreaterEqual(n_prediction_parallel_with_eval.val, N_ITEMS - 2)
        self.assertGreaterEqual(n_prediction_parallel_with_render.val, N_ITEMS - 1)
