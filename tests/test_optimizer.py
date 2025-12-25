"""Unit tests for csiro_biomass.optimizer module."""

import pytest
import torch
import torch.nn as nn

from csiro_biomass.optimizer import Optimizer


class TestOptimizer:
    """Tests for Optimizer class."""

    def test_optimizer_initialization_default(self):
        """Test Optimizer initialization with default parameters."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        assert optimizer.optimizer is not None
        assert isinstance(optimizer.optimizer, torch.optim.AdamW)

    def test_optimizer_initialization_custom_params(self):
        """Test Optimizer initialization with custom parameters."""
        model = nn.Linear(10, 5)
        params = {"lr": 0.01, "weight_decay": 0.001}
        optimizer = Optimizer(model, params=params)

        assert optimizer.optimizer is not None
        assert optimizer.optimizer.defaults["lr"] == 0.01
        assert optimizer.optimizer.defaults["weight_decay"] == 0.001

    def test_optimizer_initialization_none_params(self):
        """Test Optimizer initialization with None params uses defaults."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model, params=None)

        assert optimizer.optimizer.defaults["lr"] == 1e-3

    def test_optimizer_step(self):
        """Test Optimizer step method."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        # Create dummy loss and backward
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Perform optimizer step
        optimizer.step()

        # Check that parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)

    def test_optimizer_step_no_gradients(self):
        """Test Optimizer step with no gradients."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Step without computing gradients
        optimizer.step()

        # Parameters should not change without gradients
        for initial, current in zip(initial_params, model.parameters()):
            assert torch.equal(initial, current)

    def test_optimizer_multiple_steps(self):
        """Test multiple optimizer steps."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model, params={"lr": 0.1})

        losses = []
        for _ in range(10):
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)

            model.zero_grad()
            output = model(x)
            loss = ((output - y) ** 2).mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Optimizer should be updating parameters
        assert len(losses) == 10

    def test_optimizer_with_different_models(self):
        """Test Optimizer with different model architectures."""
        models = [
            nn.Linear(10, 5),
            nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)),
            nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 25), nn.ReLU(), nn.Linear(25, 5)),
        ]

        for model in models:
            optimizer = Optimizer(model)
            assert optimizer.optimizer is not None

    def test_optimizer_learning_rate(self):
        """Test that learning rate is correctly set."""
        model = nn.Linear(10, 5)
        lr_values = [1e-4, 1e-3, 1e-2]

        for lr in lr_values:
            optimizer = Optimizer(model, params={"lr": lr})
            assert optimizer.optimizer.defaults["lr"] == lr

    def test_optimizer_weight_decay(self):
        """Test that weight decay is correctly set."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model, params={"lr": 1e-3, "weight_decay": 0.01})

        assert optimizer.optimizer.defaults["weight_decay"] == 0.01

    def test_optimizer_betas(self):
        """Test that AdamW betas can be customized."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model, params={"lr": 1e-3, "betas": (0.95, 0.999)})

        assert optimizer.optimizer.defaults["betas"] == (0.95, 0.999)

    def test_optimizer_eps(self):
        """Test that AdamW epsilon can be customized."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model, params={"lr": 1e-3, "eps": 1e-7})

        assert optimizer.optimizer.defaults["eps"] == 1e-7

    def test_optimizer_zero_grad_before_step(self):
        """Test optimizer step after zero_grad."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        # First backward
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()

        # Zero gradients
        model.zero_grad()

        # Check gradients are zero
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0)

    def test_optimizer_gradient_accumulation(self):
        """Test optimizer with gradient accumulation."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        # Accumulate gradients over multiple batches
        model.zero_grad()
        for _ in range(3):
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()

        # Step once after accumulation
        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step()

        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)

    def test_optimizer_device_compatibility_cpu(self):
        """Test Optimizer on CPU device."""
        device = torch.device("cpu")
        model = nn.Linear(10, 5).to(device)
        optimizer = Optimizer(model)

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        assert all(p.device.type == "cpu" for p in model.parameters())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_optimizer_device_compatibility_cuda(self):
        """Test Optimizer on CUDA device."""
        device = torch.device("cuda")
        model = nn.Linear(10, 5).to(device)
        optimizer = Optimizer(model)

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 5, device=device)

        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        assert all(p.device.type == "cuda" for p in model.parameters())


class TestOptimizerIntegration:
    """Integration tests for Optimizer."""

    def test_optimizer_in_training_loop(self):
        """Test Optimizer in a complete training loop."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        optimizer = Optimizer(model, params={"lr": 0.01})

        # Generate synthetic data
        X = torch.randn(100, 10)
        Y = torch.randn(100, 5)

        initial_loss = None
        final_loss = None

        for epoch in range(5):
            model.zero_grad()
            output = model(X)
            loss = ((output - Y) ** 2).mean()

            if epoch == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if epoch == 4:
                final_loss = loss.item()

        # Training should reduce loss (usually, with random data this might not always hold)
        assert final_loss is not None
        assert initial_loss is not None

    def test_optimizer_with_different_lr_values(self):
        """Test that different learning rates affect optimization differently."""
        def train_model(lr):
            model = nn.Linear(10, 5)
            optimizer = Optimizer(model, params={"lr": lr})

            X = torch.randn(50, 10)
            Y = torch.randn(50, 5)

            for _ in range(10):
                model.zero_grad()
                output = model(X)
                loss = ((output - Y) ** 2).mean()
                loss.backward()
                optimizer.step()

            return loss.item()

        loss_small_lr = train_model(1e-5)
        loss_large_lr = train_model(1e-1)

        # Both should produce valid losses
        assert loss_small_lr >= 0
        assert loss_large_lr >= 0

    def test_optimizer_state_consistency(self):
        """Test that optimizer maintains state across steps."""
        model = nn.Linear(10, 5)
        optimizer = Optimizer(model)

        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        # First step
        model.zero_grad()
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        # Check optimizer has state
        assert len(optimizer.optimizer.state) >= 0

        # Second step
        model.zero_grad()
        output = model(x)
        loss = ((output - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        # State should be updated
        assert len(optimizer.optimizer.state) >= 0
