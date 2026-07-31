"""Triton BiGEMM forward and backward kernels.

Requires: triton >= 3.0, NVIDIA CUDA GPU.
Supported MVP shapes: g=16, d∈{64,128}, rd∈{128,256}, FP16/BF16 inputs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from omniweave.models.routing import RoutePlan

# Guard Triton import — this module must be importable without Triton installed
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


def triton_is_supported(
    x: Tensor,
    plan: RoutePlan,
    b_u: Tensor,
) -> tuple[bool, str | None]:
    """Check whether Triton execution is supported for given inputs.

    Returns ``(is_supported, reason_if_not)``.
    """
    if not _TRITON_AVAILABLE:
        return False, "Triton is not installed"

    if not x.is_cuda:
        return False, "CUDA is unavailable"

    if x.dtype not in (torch.float16, torch.bfloat16):
        return False, f"unsupported dtype {x.dtype}; expected float16 or bfloat16"

    q_s, g = plan.token_indices.shape
    if g != 16:
        return False, f"unsupported spatial tile g={g}; expected 16"

    q_c, d, rd = b_u.shape
    if d not in (64, 128):
        return False, f"unsupported channel tile d={d}; expected 64 or 128"

    if rd not in (128, 256):
        return False, f"unsupported expanded dim rd={rd}; expected 128 or 256"

    return True, None


def bigemm_delta_triton(
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
) -> Tensor:
    """Triton-accelerated BiGEMM forward pass.

    Falls back via dispatch — this is the direct kernel call.
    Requires CUDA tensors in FP16 or BF16.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")

    supported, reason = triton_is_supported(x, plan, b_u)
    if not supported:
        raise RuntimeError(f"Triton not supported: {reason}")

    # Use the autograd wrapper for forward+backward
    return _TritonBiGEMM.apply(
        x, plan, a_u, a_v, a_o, b_u, b_v, b_o, gamma,
    )


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            # num_stages=1 fits RTX 6000 Ada (99KB shared memory) for all shapes.
            # Largest dot [16,128]×[128,256] bf16 = 69KB/stage.
            triton.Config({"BLOCK_G": 16}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_G": 16}, num_warps=8, num_stages=1),
        ],
        key=["D", "RD"],
    )
    @triton.jit
    def _bigemm_fwd_kernel(
        # Packed tile input
        tiles_ptr,
        # Spatial matrices
        a_u_ptr, a_v_ptr, a_o_ptr,
        # Channel matrices
        b_u_ptr, b_v_ptr, b_o_ptr,
        # LayerScale
        gamma_ptr,
        # Mask
        mask_ptr,
        # Output
        out_ptr,
        # Dimensions
        B: tl.constexpr,
        Q_S: tl.constexpr,
        Q_C: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        RD: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """Forward kernel: one program per (batch, spatial_group, channel_group).

        All weight pointers must reference tensors pre-cast to the same dtype
        as tiles (bf16/fp16). This halves shared memory vs loading fp32 weights.
        """
        pid = tl.program_id(0)
        # Decompose pid into (b, qs, qc)
        qc = pid % Q_C
        tmp = pid // Q_C
        qs = tmp % Q_S
        b = tmp // Q_S

        # Tile offset: [B, Q_S, Q_C, G, D]
        tile_offset = (b * Q_S * Q_C + qs * Q_C + qc) * G * D
        g_range = tl.arange(0, BLOCK_G)
        d_range = tl.arange(0, D)

        # Load tile T[G, D]
        tile_offsets = tile_offset + g_range[:, None] * D + d_range[None, :]
        T = tl.load(tiles_ptr + tile_offsets)
        compute_dtype = T.dtype

        # Load mask
        mask_offset = (qs * Q_C + qc) * G * D
        mask_offsets = mask_offset + g_range[:, None] * D + d_range[None, :]
        M = tl.load(mask_ptr + mask_offsets)

        # Spatial matrices are pre-cast to compute_dtype by the wrapper
        au_offsets = g_range[:, None] * G + g_range[None, :]
        A_u = tl.load(a_u_ptr + au_offsets)

        # U branch: A_u @ T @ B_u
        # tl.dot accumulates in fp32; cast back for next dot
        U_partial = tl.dot(A_u, T).to(compute_dtype)

        rd_range = tl.arange(0, RD)
        bu_base = qc * D * RD
        bu_offsets = bu_base + d_range[:, None] * RD + rd_range[None, :]
        B_u = tl.load(b_u_ptr + bu_offsets)
        U = tl.dot(U_partial, B_u).to(compute_dtype)

        # V branch: A_v @ T @ B_v
        A_v = tl.load(a_v_ptr + au_offsets)
        V_partial = tl.dot(A_v, T).to(compute_dtype)
        bv_offsets = bu_base + d_range[:, None] * RD + rd_range[None, :]
        B_v = tl.load(b_v_ptr + bv_offsets)
        V = tl.dot(V_partial, B_v).to(compute_dtype)

        # SiLU(U) * V — compute in fp32 for numerical stability
        sigmoid_U = tl.sigmoid(U.to(tl.float32))
        silu_U = U.to(tl.float32) * sigmoid_U
        Z = (silu_U * V.to(tl.float32)).to(compute_dtype)

        # Output projection: A_o @ Z @ B_o
        A_o = tl.load(a_o_ptr + au_offsets)
        AoZ = tl.dot(A_o, Z).to(compute_dtype)

        bo_base = qc * RD * D
        bo_offsets = bo_base + rd_range[:, None] * D + d_range[None, :]
        B_o = tl.load(b_o_ptr + bo_offsets)
        delta = tl.dot(AoZ, B_o)

        # Apply mask and LayerScale
        gamma_base = qc * D
        gamma_vals = tl.load(gamma_ptr + gamma_base + d_range)
        delta = delta * gamma_vals[None, :] * M

        # Store
        tl.store(out_ptr + tile_offsets, delta)


class _TritonBiGEMM(torch.autograd.Function):
    """Custom autograd function wrapping Triton forward and backward."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        plan: RoutePlan,
        a_u: Tensor,
        a_v: Tensor,
        a_o: Tensor,
        b_u: Tensor,
        b_v: Tensor,
        b_o: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        from omniweave.ops.reference import pack_tiles, unpack_tiles

        tiles, mask = pack_tiles(x, plan)
        b, q_s, q_c, g, d = tiles.shape
        rd = b_u.shape[2]

        # Allocate output
        out = torch.zeros_like(tiles)

        # Flatten mask for kernel (remove batch broadcast)
        mask_flat = mask.squeeze(0).to(tiles.dtype)  # [Q_s, Q_c, g, d]

        # Pre-cast weights to compute dtype (tiles.dtype) so the kernel
        # loads bf16/fp16 instead of fp32. This halves shared memory usage
        # for weight buffers, critical on GPUs with ≤99KB shared memory.
        compute_dtype = tiles.dtype
        a_u_k = a_u.to(compute_dtype).contiguous()
        a_v_k = a_v.to(compute_dtype).contiguous()
        a_o_k = a_o.to(compute_dtype).contiguous()
        b_u_k = b_u.to(compute_dtype).contiguous()
        b_v_k = b_v.to(compute_dtype).contiguous()
        b_o_k = b_o.to(compute_dtype).contiguous()

        # Reshape gamma to match channel groups and pre-cast
        ch_idx = plan.channel_indices.clamp_max(plan.channels - 1).to(gamma.device)
        ch_mask = plan.channel_mask.to(gamma.device, gamma.dtype)
        grouped_gamma = (gamma[ch_idx] * ch_mask).to(compute_dtype).contiguous()

        # Launch kernel
        grid = (b * q_s * q_c,)
        _bigemm_fwd_kernel[grid](
            tiles.contiguous(),
            a_u_k, a_v_k, a_o_k,
            b_u_k, b_v_k, b_o_k,
            grouped_gamma,
            mask_flat.contiguous(),
            out,
            B=b, Q_S=q_s, Q_C=q_c, G=g, D=d, RD=rd,
        )

        # Save for backward
        ctx.save_for_backward(x, tiles, mask.to(tiles.dtype), a_u, a_v, a_o, b_u, b_v, b_o, gamma)
        ctx.plan = plan

        return unpack_tiles(out, plan, b)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Backward pass — fall back to reference for correctness.

        All computation is done in fp32 to avoid mixed-dtype issues
        (AMP saves x as bf16 but weights as fp32) and for gradient accuracy.
        """
        from omniweave.ops.reference import bigemm_delta_reference

        x, tiles, mask, a_u, a_v, a_o, b_u, b_v, b_o, gamma = ctx.saved_tensors
        plan = ctx.plan

        # Upcast everything to fp32 for uniform dtype in einsum
        x_ref = x.detach().float().requires_grad_(True)
        a_u_ref = a_u.detach().float().requires_grad_(True)
        a_v_ref = a_v.detach().float().requires_grad_(True)
        a_o_ref = a_o.detach().float().requires_grad_(True)
        b_u_ref = b_u.detach().float().requires_grad_(True)
        b_v_ref = b_v.detach().float().requires_grad_(True)
        b_o_ref = b_o.detach().float().requires_grad_(True)
        gamma_ref = gamma.detach().float().requires_grad_(True)

        with torch.enable_grad():
            out = bigemm_delta_reference(
                x=x_ref, plan=plan,
                a_u=a_u_ref, a_v=a_v_ref, a_o=a_o_ref,
                b_u=b_u_ref, b_v=b_v_ref, b_o=b_o_ref,
                gamma=gamma_ref,
            )
            out.backward(grad_output.float())

        return (
            x_ref.grad.to(x.dtype) if x_ref.grad is not None else None,
            None,  # plan
            a_u_ref.grad, a_v_ref.grad, a_o_ref.grad,
            b_u_ref.grad, b_v_ref.grad, b_o_ref.grad,
            gamma_ref.grad,
        )

if not _TRITON_AVAILABLE:
    # Triton not available — provide a stub class
    class _TritonBiGEMM:  # type: ignore[no-redef]
        @staticmethod
        def apply(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Triton is not installed")
