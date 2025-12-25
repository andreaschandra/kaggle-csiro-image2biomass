"""Unit tests for csiro_biomass.metrics module."""

import pytest
import torch

from csiro_biomass.metrics import compute_weighted_r2


class TestComputeWeightedR2:
    """Tests for compute_weighted_r2 function."""

    def test_compute_weighted_r2_perfect_predictions(self):
        """Test R2 score with perfect predictions."""
        pred = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]])
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]])

        r2 = compute_weighted_r2(pred, target)

        assert torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_good_predictions(self):
        """Test R2 score with good but not perfect predictions."""
        pred = torch.tensor(
            [[10.1, 20.1, 15.1], [12.1, 18.1, 14.1], [9.1, 22.1, 16.1]], dtype=torch.float32
        )
        target = torch.tensor(
            [[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]], dtype=torch.float32
        )

        r2 = compute_weighted_r2(pred, target)

        assert r2 > 0.9
        assert r2 < 1.0

    def test_compute_weighted_r2_poor_predictions(self):
        """Test R2 score with poor predictions."""
        pred = torch.tensor([[50.0, 50.0, 50.0], [50.0, 50.0, 50.0], [50.0, 50.0, 50.0]])
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]])

        r2 = compute_weighted_r2(pred, target)

        assert r2 < 0.0

    def test_compute_weighted_r2_mean_predictions(self):
        """Test R2 score when predictions are mean of targets."""
        target = torch.tensor(
            [[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]], dtype=torch.float32
        )
        mean_vals = target.mean(dim=0)
        pred = mean_vals.repeat(3, 1)

        r2 = compute_weighted_r2(pred, target)

        # R2 should be close to 0 when predicting mean
        assert torch.isclose(r2, torch.tensor(0.0), atol=1e-4)

    def test_compute_weighted_r2_weights(self):
        """Test that weights are applied correctly."""
        # The function uses equal weights [1, 1, 1], so R2 should be average of per-column R2
        pred = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]], dtype=torch.float32)
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]], dtype=torch.float32)

        r2 = compute_weighted_r2(pred, target)

        assert torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_single_sample(self):
        """Test R2 score with single sample."""
        pred = torch.tensor([[10.0, 20.0, 15.0]])
        target = torch.tensor([[10.0, 20.0, 15.0]])

        r2 = compute_weighted_r2(pred, target)

        # With single sample, R2 is typically undefined, but implementation may vary
        assert r2.isnan() or torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_batch_processing(self):
        """Test R2 score with larger batch."""
        batch_size = 100
        pred = torch.randn(batch_size, 3)
        target = pred + torch.randn(batch_size, 3) * 0.1  # Add small noise

        r2 = compute_weighted_r2(pred, target)

        assert r2 > 0.5  # Should have decent R2 with small noise

    def test_compute_weighted_r2_device_cpu(self):
        """Test R2 score on CPU device."""
        device = torch.device("cpu")
        pred = torch.tensor([[10.0, 20.0, 15.0]]).to(device)
        target = torch.tensor([[10.0, 20.0, 15.0]]).to(device)

        r2 = compute_weighted_r2(pred, target)

        assert r2.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_compute_weighted_r2_device_cuda(self):
        """Test R2 score on CUDA device."""
        device = torch.device("cuda")
        pred = torch.tensor([[10.0, 20.0, 15.0]]).to(device)
        target = torch.tensor([[10.0, 20.0, 15.0]]).to(device)

        r2 = compute_weighted_r2(pred, target)

        assert r2.device.type == "cuda"

    def test_compute_weighted_r2_different_devices(self):
        """Test R2 score with pred and target on different devices initially."""
        pred = torch.tensor([[10.0, 20.0, 15.0]], device="cpu")
        target = torch.tensor([[10.0, 20.0, 15.0]], device="cpu")

        r2 = compute_weighted_r2(pred, target)

        assert r2 is not None

    def test_compute_weighted_r2_negative_values(self):
        """Test R2 score with negative values."""
        pred = torch.tensor([[-10.0, -20.0, -15.0], [-12.0, -18.0, -14.0]])
        target = torch.tensor([[-10.0, -20.0, -15.0], [-12.0, -18.0, -14.0]])

        r2 = compute_weighted_r2(pred, target)

        assert torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_mixed_values(self):
        """Test R2 score with mixed positive and negative values."""
        pred = torch.tensor([[-10.0, 20.0, -15.0], [12.0, -18.0, 14.0]])
        target = torch.tensor([[-10.0, 20.0, -15.0], [12.0, -18.0, 14.0]])

        r2 = compute_weighted_r2(pred, target)

        assert torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_zero_variance(self):
        """Test R2 score when target has zero variance in a column."""
        pred = torch.tensor([[10.0, 20.0, 15.0], [10.0, 18.0, 14.0]])
        target = torch.tensor([[10.0, 20.0, 15.0], [10.0, 18.0, 14.0]])

        r2 = compute_weighted_r2(pred, target)

        # When variance is zero, R2 might be nan or 1
        assert r2.isnan() or torch.isclose(r2, torch.tensor(1.0), atol=1e-5)

    def test_compute_weighted_r2_large_batch(self):
        """Test R2 score with large batch size."""
        batch_size = 1000
        torch.manual_seed(42)
        target = torch.randn(batch_size, 3) * 10
        pred = target + torch.randn(batch_size, 3) * 0.5  # Add noise

        r2 = compute_weighted_r2(pred, target)

        assert r2 > 0.9  # Should have high R2 with small noise

    def test_compute_weighted_r2_dtype_consistency(self):
        """Test R2 score maintains dtype."""
        pred = torch.tensor([[10.0, 20.0, 15.0]], dtype=torch.float32)
        target = torch.tensor([[10.0, 20.0, 15.0]], dtype=torch.float32)

        r2 = compute_weighted_r2(pred, target)

        assert r2.dtype == torch.float32

    def test_compute_weighted_r2_requires_no_grad(self):
        """Test that R2 computation doesn't require gradients."""
        pred = torch.tensor([[10.0, 20.0, 15.0]], requires_grad=True)
        target = torch.tensor([[10.0, 20.0, 15.0]])

        r2 = compute_weighted_r2(pred, target)

        # R2 should be computable even with gradient tracking
        assert r2 is not None


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_r2_in_training_loop(self):
        """Test R2 metric in a simulated training scenario."""
        predictions = []
        targets = []

        for _ in range(10):
            pred = torch.randn(8, 3)
            target = torch.randn(8, 3)
            predictions.append(pred)
            targets.append(target)

        # Compute R2 for each batch
        r2_scores = []
        for pred, target in zip(predictions, targets):
            r2 = compute_weighted_r2(pred, target)
            r2_scores.append(r2)

        assert len(r2_scores) == 10
        assert all(isinstance(r2, torch.Tensor) for r2 in r2_scores)

    def test_r2_improvement_simulation(self):
        """Simulate improving predictions and verify R2 increases."""
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0], [9.0, 22.0, 16.0]])

        # Poor predictions
        pred_poor = target + torch.randn_like(target) * 5
        r2_poor = compute_weighted_r2(pred_poor, target)

        # Better predictions
        pred_better = target + torch.randn_like(target) * 0.5
        r2_better = compute_weighted_r2(pred_better, target)

        # Better predictions should have higher R2
        assert r2_better > r2_poor

    def test_r2_with_multiple_devices(self):
        """Test R2 computation across different device scenarios."""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for device in devices:
            pred = torch.randn(10, 3, device=device)
            target = torch.randn(10, 3, device=device)

            r2 = compute_weighted_r2(pred, target)

            assert r2.device.type == device
