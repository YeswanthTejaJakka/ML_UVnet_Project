"""
P1 — Numerical correctness tests for PScan and PVMLayer.
Tests:
1. PScan forward against sequential reference scan.
2. PScan backward with torch.autograd.gradcheck (in float64, with non-power-of-2 L).
3. Batched PVMLayer output matching sequential reference implementation.
4. Parameter count check for inference model.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import torch
from models.ultralight_vmunet import PScan, pscan, PVMLayer, UltraLight_VM_UNet


def sequential_scan(dA, Bx):
    """
    Reference sequential scan: h_t = dA_t * h_{t-1} + Bx_t
    dA, Bx: [B, L, D, N]
    Returns: [B, L, D, N]
    """
    B, L, D, N = dA.shape
    h = torch.zeros(B, D, N, dtype=dA.dtype, device=dA.device)
    output = []
    for t in range(L):
        h = dA[:, t] * h + Bx[:, t]
        output.append(h.unsqueeze(1))
    return torch.cat(output, dim=1)


def test_pscan_forward():
    """PScan forward output must match sequential scan within numerical tolerance."""
    for L in [4, 8, 16, 24, 32, 64, 1000]:
        B, D, N = 2, 4, 8
        torch.manual_seed(42)
        dA = torch.rand(B, L, D, N, dtype=torch.float32) * 0.9 + 0.05
        Bx = torch.randn(B, L, D, N, dtype=torch.float32) * 0.1

        ref = sequential_scan(dA, Bx)
        out = pscan(dA, Bx)

        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        max_diff = (out - ref).abs().max().item()
        assert max_diff < 1e-4, f"Forward max diff {max_diff:.2e} at L={L}"
    print("[PASS] test_pscan_forward across all sequence lengths (including L=24, 1000).")


def test_pscan_gradcheck():
    """PScan autograd backward pass check via gradcheck in float64."""
    for L in [4, 8, 16, 24]:
        B, D, N = 1, 2, 4
        torch.manual_seed(42)
        dA = (torch.rand(B, L, D, N, dtype=torch.float64) * 0.5 + 0.3).requires_grad_(True)
        Bx = (torch.randn(B, L, D, N, dtype=torch.float64) * 0.05).requires_grad_(True)

        result = torch.autograd.gradcheck(
            pscan,
            (dA, Bx),
            eps=1e-5,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )
        assert result, f"gradcheck failed for L={L}"
    print("[PASS] test_pscan_gradcheck in float64.")


def test_pvm_batched_vs_sequential():
    """
    P1 requirement: PVMLayer batched vs 4 sequential calls.
    Output must match to within 1e-5.
    """
    torch.manual_seed(42)
    B, C, H, W = 2, 24, 16, 16
    layer = PVMLayer(input_dim=C, output_dim=C)
    layer.eval()

    x = torch.randn(B, C, H, W)

    # Batched implementation
    with torch.no_grad():
        out_batched = layer(x)

    # Sequential reference implementation
    with torch.no_grad():
        n_tokens = H * W
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = layer.norm(x_flat)
        chunk_size = C // 4
        x1, x2, x3, x4 = x_norm.split(chunk_size, dim=-1)

        xm1 = layer.mamba(x1) + layer.skip_scale * x1
        xm2 = layer.mamba(x2) + layer.skip_scale * x2
        xm3 = layer.mamba(x3) + layer.skip_scale * x3
        xm4 = layer.mamba(x4) + layer.skip_scale * x4
        x_seq = torch.cat([xm1, xm2, xm3, xm4], dim=-1)
        x_seq = layer.norm(x_seq)
        x_seq = layer.proj(x_seq)
        out_sequential = x_seq.transpose(-1, -2).reshape(B, C, H, W)

    max_diff = (out_batched - out_sequential).abs().max().item()
    assert max_diff < 1e-5, f"PVMLayer batched vs sequential difference too large: {max_diff:.2e}"
    print(f"[PASS] test_pvm_batched_vs_sequential (max diff = {max_diff:.2e} < 1e-5).")


def test_inference_param_count():
    """UltraLight VM-UNet inference parameter count must be ~0.0495M."""
    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att='fc',
        bridge=True,
        aux_boundary_head=False,
        deep_supervision=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert 45_000 < n_params < 55_000, f"Unexpected parameter count: {n_params}"
    assert round(n_params / 1e6, 4) == 0.0495, f"Expected 0.0495M params, got {n_params / 1e6:.4f}M"
    print(f"[PASS] test_inference_param_count: {n_params} parameters ({n_params/1e6:.4f}M).")


if __name__ == "__main__":
    print("Running PScan & PVMLayer correctness suite...")
    test_pscan_forward()
    test_pscan_gradcheck()
    test_pvm_batched_vs_sequential()
    test_inference_param_count()
    print("ALL P1 TESTS PASSED SUCCESSFULLY!")

