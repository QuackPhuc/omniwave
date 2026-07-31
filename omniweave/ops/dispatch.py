"""Backend dispatch — selects reference, compile, or Triton execution."""

from __future__ import annotations

import logging
from collections.abc import Callable

from torch import Tensor

logger = logging.getLogger(__name__)


def select_backend(
    x: Tensor,
    requested: str,
    support_check: Callable[[Tensor], tuple[bool, str | None]],
) -> tuple[str, str | None]:
    """Select execution backend with automatic fallback.

    Parameters
    ----------
    x : sample input tensor (used for device/dtype checks)
    requested : ``"reference"`` | ``"compile"`` | ``"triton"`` | ``"auto"``
    support_check : callable returning ``(is_supported, reason_if_not)``

    Returns
    -------
    (backend, reason) : chosen backend name and fallback reason (None if no fallback)

    Raises
    ------
    RuntimeError
        If forced ``"triton"`` mode is unsupported.
    """
    if requested not in {"reference", "compile", "triton", "auto"}:
        raise ValueError(f"unknown backend: {requested!r}")

    if requested in {"reference", "compile"}:
        return requested, None

    supported, reason = support_check(x)
    if supported:
        return "triton", None

    if requested == "triton":
        raise RuntimeError(reason)

    logger.info("falling back to reference backend: %s", reason)
    return "reference", reason
