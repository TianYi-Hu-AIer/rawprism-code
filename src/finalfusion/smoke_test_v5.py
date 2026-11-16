"""CPU/GPU smoke test for the v5 components (no data required)."""

from __future__ import annotations

import torch

from finalfusion.v5_components import (
    DifferentiableRawRenderer,
    PhysicsAnchoredFusion,
    ReliabilityFieldTransformer,
    physical_reliability_priors,
)


def main() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 64, 64
    raw1 = torch.rand(b, 4, h, w)
    raw2 = torch.rand(b, 4, h, w)

    priors = physical_reliability_priors(raw1, raw2, "mef", {"a1": torch.ones(1, 4, 1, 1) * 0.1,
                                                              "b1": torch.ones(1, 4, 1, 1) * 1e-4,
                                                              "a2": torch.ones(1, 4, 1, 1) * 0.1,
                                                              "b2": torch.ones(1, 4, 1, 1) * 1e-4})
    assert priors.shape == (b, 8, h, w), priors.shape

    z1 = torch.rand(b, 6, h, w)
    z2 = torch.rand(b, 6, h, w)
    cond = torch.randn(b, 32)
    rft = ReliabilityFieldTransformer(use_semantic=False)
    w1, w2 = rft(priors, z1, z2, cond)
    assert w1.shape == w2.shape == (b, 1, h, w)
    assert torch.allclose(w1 + w2, torch.ones_like(w1), atol=1e-5)

    fusion = PhysicsAnchoredFusion()
    u = fusion(z1, z2, w1, w2)
    assert u.shape == (b, 6, h, w)

    renderer = DifferentiableRawRenderer()
    fused_raw = torch.rand(b, 4, h, w)
    srgb = renderer(fused_raw)
    assert srgb.shape == (b, 3, h, w)
    assert srgb.min() >= 0.0 and srgb.max() <= 1.0

    print("v5 smoke test passed.")


if __name__ == "__main__":
    main()

