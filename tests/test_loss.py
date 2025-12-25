"""Unit tests for csiro_biomass.loss module."""

import pytest
import torch

from csiro_biomass.loss import TweedieLoss, WeightedHuberLoss, WeightedMSELoss


class TestWeightedMSELoss:
    """Tests for WeightedMSELoss."""

    def test_weighted_mse_loss_initialization(self):
        """Test WeightedMSELoss initialization."""
        loss_fn = WeightedMSELoss()

        assert loss_fn.weights is not None
        assert len(loss_fn.weights) == 5
        assert loss_fn.weights.tolist() == [0.1, 0.1, 0.1, 0.5, 0.2]

    def test_weighted_mse_loss_forward(self):
        """Test WeightedMSELoss forward pass."""
        loss_fn = WeightedMSELoss()
        pred = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0], [12.0, 18.0, 14.0, 44.0, 28.0]])
        target = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0], [12.0, 18.0, 14.0, 44.0, 28.0]])

        loss = loss_fn(pred, target)

        assert loss.item() == 0.0

    def test_weighted_mse_loss_non_zero(self):
        """Test WeightedMSELoss with non-zero loss."""
        loss_fn = WeightedMSELoss()
        pred = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0]])
        target = torch.tensor([[11.0, 21.0, 16.0, 46.0, 31.0]])

        loss = loss_fn(pred, target)

        assert loss.item() > 0.0

    def test_weighted_mse_loss_weights_applied(self):
        """Test that weights are correctly applied."""
        loss_fn = WeightedMSELoss()
        # Create predictions with error only in high-weight column (index 3, weight 0.5)
        pred = torch.tensor([[10.0, 20.0, 15.0, 50.0, 30.0]])
        target = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0]])

        loss_high_weight = loss_fn(pred, target)

        # Create predictions with same error in low-weight column (index 0, weight 0.1)
        pred2 = torch.tensor([[15.0, 20.0, 15.0, 45.0, 30.0]])
        target2 = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0]])

        loss_low_weight = loss_fn(pred2, target2)

        # Error in high-weight column should produce higher loss
        assert loss_high_weight.item() > loss_low_weight.item()

    def test_weighted_mse_loss_batch_processing(self):
        """Test WeightedMSELoss with batch input."""
        loss_fn = WeightedMSELoss()
        batch_size = 16
        pred = torch.randn(batch_size, 5)
        target = torch.randn(batch_size, 5)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar output
        assert loss.item() >= 0.0

    def test_weighted_mse_loss_device_compatibility(self):
        """Test WeightedMSELoss device compatibility."""
        loss_fn = WeightedMSELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = torch.randn(4, 5).to(device)
        target = torch.randn(4, 5).to(device)

        loss = loss_fn(pred, target)

        assert loss.device.type in ["cuda", "cpu"]

    def test_weighted_mse_loss_gradient_flow(self):
        """Test that gradients flow through WeightedMSELoss."""
        loss_fn = WeightedMSELoss()
        pred = torch.randn(4, 5, requires_grad=True)
        target = torch.randn(4, 5)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestWeightedHuberLoss:
    """Tests for WeightedHuberLoss."""

    def test_weighted_huber_loss_initialization(self):
        """Test WeightedHuberLoss initialization."""
        loss_fn = WeightedHuberLoss(delta=30, device="cpu")

        assert loss_fn.delta == 30
        assert len(loss_fn.weights) == 3
        assert loss_fn.weights.tolist() == [0.3, 0.3, 0.4]

    def test_weighted_huber_loss_custom_delta(self):
        """Test WeightedHuberLoss with custom delta."""
        loss_fn = WeightedHuberLoss(delta=10, device="cpu")

        assert loss_fn.delta == 10

    def test_weighted_huber_loss_forward(self):
        """Test WeightedHuberLoss forward pass."""
        loss_fn = WeightedHuberLoss(delta=3.0, device="cpu")
        pred = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]])
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]])

        loss = loss_fn(pred, target)

        assert loss.item() == 0.0

    def test_weighted_huber_loss_small_error(self):
        """Test WeightedHuberLoss with small error (quadratic region)."""
        loss_fn = WeightedHuberLoss(delta=5.0, device="cpu")
        pred = torch.tensor([[10.0, 20.0, 15.0]])
        target = torch.tensor([[11.0, 21.0, 16.0]])  # Error = 1 < delta

        loss = loss_fn(pred, target)

        assert loss.item() > 0.0

    def test_weighted_huber_loss_large_error(self):
        """Test WeightedHuberLoss with large error (linear region)."""
        loss_fn = WeightedHuberLoss(delta=1.0, device="cpu")
        pred = torch.tensor([[10.0, 20.0, 15.0]])
        target = torch.tensor([[15.0, 25.0, 20.0]])  # Error = 5 > delta

        loss = loss_fn(pred, target)

        assert loss.item() > 0.0

    def test_weighted_huber_loss_weights_applied(self):
        """Test that weights are correctly applied."""
        loss_fn = WeightedHuberLoss(delta=10.0, device="cpu")
        # Error in high-weight column (index 2, weight 0.4)
        pred = torch.tensor([[10.0, 20.0, 20.0]])
        target = torch.tensor([[10.0, 20.0, 15.0]])

        loss_high_weight = loss_fn(pred, target)

        # Same error in low-weight column (index 0, weight 0.3)
        pred2 = torch.tensor([[15.0, 20.0, 15.0]])
        target2 = torch.tensor([[10.0, 20.0, 15.0]])

        loss_low_weight = loss_fn(pred2, target2)

        # Error in high-weight column should produce higher loss
        assert loss_high_weight.item() > loss_low_weight.item()

    def test_weighted_huber_loss_batch_processing(self):
        """Test WeightedHuberLoss with batch input."""
        loss_fn = WeightedHuberLoss(delta=3.0, device="cpu")
        batch_size = 16
        pred = torch.randn(batch_size, 3)
        target = torch.randn(batch_size, 3)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_weighted_huber_loss_device_compatibility(self):
        """Test WeightedHuberLoss device compatibility."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = WeightedHuberLoss(delta=3.0, device=str(device))
        pred = torch.randn(4, 3).to(device)
        target = torch.randn(4, 3).to(device)

        loss = loss_fn(pred, target)

        assert loss.device.type in ["cuda", "cpu"]

    def test_weighted_huber_loss_gradient_flow(self):
        """Test that gradients flow through WeightedHuberLoss."""
        loss_fn = WeightedHuberLoss(delta=3.0, device="cpu")
        pred = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None


class TestTweedieLoss:
    """Tests for TweedieLoss."""

    def test_tweedie_loss_initialization(self):
        """Test TweedieLoss initialization."""
        loss_fn = TweedieLoss(p=1.5)

        assert loss_fn.p == 1.5

    def test_tweedie_loss_invalid_p_too_low(self):
        """Test TweedieLoss with invalid p value (too low)."""
        with pytest.raises(AssertionError):
            TweedieLoss(p=0.5)

    def test_tweedie_loss_invalid_p_too_high(self):
        """Test TweedieLoss with invalid p value (too high)."""
        with pytest.raises(AssertionError):
            TweedieLoss(p=2.5)

    def test_tweedie_loss_invalid_p_boundary_low(self):
        """Test TweedieLoss with p at lower boundary."""
        with pytest.raises(AssertionError):
            TweedieLoss(p=1.0)

    def test_tweedie_loss_invalid_p_boundary_high(self):
        """Test TweedieLoss with p at upper boundary."""
        with pytest.raises(AssertionError):
            TweedieLoss(p=2.0)

    def test_tweedie_loss_valid_p_values(self):
        """Test TweedieLoss with valid p values."""
        for p in [1.1, 1.5, 1.9]:
            loss_fn = TweedieLoss(p=p)
            assert loss_fn.p == p

    def test_tweedie_loss_forward(self):
        """Test TweedieLoss forward pass."""
        loss_fn = TweedieLoss(p=1.5)
        pred = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]])
        target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 18.0, 14.0]])

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_tweedie_loss_positive_predictions(self):
        """Test TweedieLoss handles negative predictions (clamps to positive)."""
        loss_fn = TweedieLoss(p=1.5)
        pred = torch.tensor([[-5.0, 10.0, 20.0]])
        target = torch.tensor([[10.0, 10.0, 20.0]])

        loss = loss_fn(pred, target)

        # Should not raise error; negative predictions are clamped
        assert loss.item() >= 0.0

    def test_tweedie_loss_non_zero(self):
        """Test TweedieLoss with non-matching predictions."""
        loss_fn = TweedieLoss(p=1.5)
        pred = torch.tensor([[10.0, 20.0, 15.0]])
        target = torch.tensor([[12.0, 22.0, 17.0]])

        loss = loss_fn(pred, target)

        assert loss.item() != 0.0

    def test_tweedie_loss_batch_processing(self):
        """Test TweedieLoss with batch input."""
        loss_fn = TweedieLoss(p=1.5)
        batch_size = 16
        pred = torch.abs(torch.randn(batch_size, 3)) + 1.0  # Ensure positive
        target = torch.abs(torch.randn(batch_size, 3)) + 1.0

        loss = loss_fn(pred, target)

        assert loss.dim() == 0

    def test_tweedie_loss_different_p_values(self):
        """Test TweedieLoss behavior with different p values."""
        pred = torch.tensor([[10.0, 20.0, 15.0]])
        target = torch.tensor([[12.0, 22.0, 17.0]])

        loss_1_2 = TweedieLoss(p=1.2)(pred, target)
        loss_1_5 = TweedieLoss(p=1.5)(pred, target)
        loss_1_8 = TweedieLoss(p=1.8)(pred, target)

        # Different p values should give different losses
        assert loss_1_2.item() != loss_1_5.item()
        assert loss_1_5.item() != loss_1_8.item()

    def test_tweedie_loss_gradient_flow(self):
        """Test that gradients flow through TweedieLoss."""
        loss_fn = TweedieLoss(p=1.5)
        pred = torch.abs(torch.randn(4, 3, requires_grad=True)) + 1.0
        target = torch.abs(torch.randn(4, 3)) + 1.0

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_tweedie_loss_device_compatibility(self):
        """Test TweedieLoss device compatibility."""
        loss_fn = TweedieLoss(p=1.5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = (torch.abs(torch.randn(4, 3)) + 1.0).to(device)
        target = (torch.abs(torch.randn(4, 3)) + 1.0).to(device)

        loss = loss_fn(pred, target)

        assert loss.device.type in ["cuda", "cpu"]


class TestLossIntegration:
    """Integration tests for loss functions."""

    def test_all_losses_with_same_input(self):
        """Test all loss functions with the same input."""
        pred = torch.tensor([[10.0, 20.0, 15.0, 45.0, 30.0]])
        target = torch.tensor([[12.0, 22.0, 17.0, 47.0, 32.0]])

        mse_loss = WeightedMSELoss()(pred, target)
        huber_loss = WeightedHuberLoss(delta=3.0, device="cpu")(pred[:, :3], target[:, :3])
        tweedie_loss = TweedieLoss(p=1.5)(pred[:, :3], target[:, :3])

        assert all(loss.item() >= 0.0 for loss in [mse_loss, huber_loss, tweedie_loss])

    def test_loss_in_training_loop(self):
        """Test loss functions in a simulated training loop."""
        loss_fn = WeightedMSELoss()
        model_output = torch.randn(8, 5, requires_grad=True)
        target = torch.randn(8, 5)

        loss = loss_fn(model_output, target)
        loss.backward()

        assert model_output.grad is not None
        assert loss.item() >= 0.0
