"""The KV-cache slice for Moondream2's decode step (`moondream.sliced_decode_attn`), checked on a toy `attn` with
the pinned remote code's signature, cache and mask layout, so no model is downloaded."""

from typing import Any

import torch
from torch.nn import functional as F

from fusion_vision_mcp.moondream import sliced_decode_attn

HEADS, DIM, CONTEXT = 4, 8, 64


class Cache:
    """The pinned `KVCache`: a preallocated buffer, written at `pos_ids`, returned whole."""

    def __init__(self) -> None:
        self.k_cache = torch.zeros(1, HEADS, CONTEXT, DIM)
        self.v_cache = torch.zeros(1, HEADS, CONTEXT, DIM)

    def update(self, pos_ids: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k_cache[:, :, pos_ids, :] = k
        self.v_cache[:, :, pos_ids, :] = v
        return self.k_cache, self.v_cache


def toy_attn(
    x: torch.Tensor,
    w: Any,
    freqs_cis: Any,
    kv_cache: Any,
    attn_mask: Any,
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
    lora: Any,
) -> torch.Tensor:
    q = k = v = x.view(1, x.shape[1], HEADS, DIM).transpose(1, 2)
    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def _filled(pos: int) -> tuple[Cache, Cache]:
    a, b = Cache(), Cache()
    prefix = torch.randn(1, HEADS, pos, DIM)
    for c in (a, b):
        c.k_cache[:, :, :pos], c.v_cache[:, :, :pos] = prefix, prefix.clone()
    return a, b


def _decode(attn: Any, cache: Cache, x: torch.Tensor, pos: int, mask: torch.Tensor) -> torch.Tensor:
    return attn(x, None, None, cache, mask, HEADS, HEADS, torch.tensor([pos]), None)


def test_decode_step_matches_the_masked_full_cache() -> None:
    torch.manual_seed(0)
    pos = 20
    full, sliced = _filled(pos)
    mask = torch.zeros(1, 1, CONTEXT, dtype=torch.bool)
    mask[:, :, : pos + 1] = True  # what the pinned generator builds: `:pos` set, then `pos`
    x = torch.randn(1, 1, HEADS * DIM)
    expected = _decode(toy_attn, full, x, pos, mask)
    got = _decode(sliced_decode_attn(toy_attn), sliced, x, pos, mask)
    torch.testing.assert_close(got, expected)
    # The new key still lands in the real buffer, so the next step sees it.
    torch.testing.assert_close(sliced.k_cache, full.k_cache)


def test_steps_with_another_mask_run_unchanged() -> None:
    calls: list[tuple[Any, Any]] = []

    def spy(x: Any, w: Any, f: Any, kv_cache: Any, attn_mask: Any, *rest: Any) -> torch.Tensor:
        calls.append((kv_cache, attn_mask))
        return x

    attn = sliced_decode_attn(spy)
    cache = Cache()
    x = torch.zeros(1, 1, HEADS * DIM)
    beyond = torch.ones(1, 1, CONTEXT, dtype=torch.bool)  # true past pos
    attn(x, None, None, cache, beyond, HEADS, HEADS, torch.tensor([9]), None)
    prefill = torch.ones(1, 1, 3, CONTEXT, dtype=torch.bool)
    attn(torch.zeros(1, 3, HEADS * DIM), None, None, cache, prefill, HEADS, HEADS, torch.arange(3), None)
    assert [(c is cache, m is mask) for (c, m), mask in zip(calls, (beyond, prefill), strict=True)] == [
        (True, True)
    ] * 2


def test_decode_step_keeps_a_sliced_mask() -> None:
    # Dropping the mask changes CPU SDPA's kernel and its rounding; a gap inside the prefix must survive too.
    seen: list[Any] = []
    attn = sliced_decode_attn(lambda x, w, f, kv, m, *rest: seen.append(m) or x)
    holed = torch.zeros(1, 1, CONTEXT, dtype=torch.bool)
    holed[:, :, :5] = True
    holed[:, :, 9] = True
    attn(torch.zeros(1, 1, HEADS * DIM), None, None, Cache(), holed, HEADS, HEADS, torch.tensor([9]), None)
    assert seen[0].shape[-1] == 10
    assert torch.equal(seen[0], holed[..., :10])
