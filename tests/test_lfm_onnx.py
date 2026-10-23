"""The default captioner's sampling step (`lfm_onnx.pick_token`), checked without the model."""

import numpy as np

from fusion_vision_mcp.lfm_onnx import DECODING, pick_token


def test_temperature_zero_is_argmax() -> None:
    logits = np.array([0.1, 2.0, 1.9, -1.0])
    assert pick_token(logits, np.array([], np.int64), {"temperature": 0.0}, np.random.default_rng(0)) == 1


def test_repetition_penalty_divides_positive_and_multiplies_negative_logits() -> None:
    # Token 1 leads by less than the penalty takes off it, so a seen token 1 loses to token 2.
    logits = np.array([-5.0, 2.0, 1.95, -5.0])
    greedy = {"temperature": 0.0, "repeat_penalty": 1.05}
    assert pick_token(logits, np.array([1]), greedy, np.random.default_rng(0)) == 2
    # A negative logit is pushed further down, not up.
    logits = np.array([-1.0, -1.02, -9.0])
    assert pick_token(logits, np.array([0]), greedy, np.random.default_rng(0)) == 1


def test_min_p_keeps_only_tokens_near_the_top() -> None:
    # At temperature 0.1 a 1-logit gap is a factor of e^10, far under min_p 0.15: only the top token survives.
    logits = np.array([3.0, 2.0, 2.0, 0.0])
    rng = np.random.default_rng(123)
    assert {pick_token(logits, np.array([], np.int64), DECODING, rng) for _ in range(50)} == {0}


def test_same_seed_same_choices() -> None:
    # Two near-tied tokens, so sampling really chooses; one seed must repeat its choices exactly.
    logits = np.array([1.0, 1.0, 0.99, -3.0])
    decoding = {"temperature": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}

    def draws(seed: int) -> list[int]:
        rng = np.random.default_rng(seed)
        return [pick_token(logits, np.array([], np.int64), decoding, rng) for _ in range(20)]

    assert draws(7) == draws(7)
    assert len(set(draws(7))) > 1
