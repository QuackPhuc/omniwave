"""Public BiGEMM entry point — delegates to backend implementations."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from omniweave.models.routing import RoutePlan
from omniweave.ops.reference import bigemm_delta_reference

_compiled_fn: Any = None


def _get_compiled_fn() -> Any:
    """Lazily compile the reference implementation once per process."""
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(bigemm_delta_reference, dynamic=False)
    return _compiled_fn


def bigemm(
    *,
    x: Tensor,
    plan: RoutePlan,
    a_u: Tensor,
    a_v: Tensor,
    a_o: Tensor,
    b_u: Tensor,
    b_v: Tensor,
    b_o: Tensor,
    gamma: Tensor,
    backend: str = "reference",
) -> Tensor:
    """Compute the BiGEMM residual delta.

    Parameters
    ----------
    backend : ``"reference"`` | ``"compile"`` | ``"triton"`` | ``"auto"``
    """
    kwargs = dict(
        x=x, plan=plan,
        a_u=a_u, a_v=a_v, a_o=a_o,
        b_u=b_u, b_v=b_v, b_o=b_o,
        gamma=gamma,
    )

    if backend == "reference":
        return bigemm_delta_reference(**kwargs)

    if backend == "compile":
        return _get_compiled_fn()(**kwargs)

    if backend == "triton":
        from omniweave.ops.triton import bigemm_delta_triton
        return bigemm_delta_triton(**kwargs)

    if backend == "auto":
        from omniweave.ops.dispatch import select_backend
        from omniweave.ops.triton import triton_is_supported

        chosen, reason = select_backend(
            x, "auto",
            lambda t: triton_is_supported(t, plan, b_u),
        )
        return bigemm(backend=chosen, **kwargs)

    raise ValueError(
        f"unknown backend: {backend!r}; "
        f"expected 'reference', 'compile', 'triton', or 'auto'"
    )
