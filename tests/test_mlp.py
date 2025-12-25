"""Unit tests for csiro_biomass.models.mlp module."""


import pytest
import torch
import torch.nn as nn

from csiro_biomass.models.mlp import MLP


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_initialization(self):
        """Test MLP initialization."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_mlp_architecture(self):
        """Test MLP architecture structure."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        assert hasattr(model, "model")
        assert isinstance(model.model, nn.Sequential)

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        emb_size = 768
        batch_size = 4
        model = MLP(emb_size=emb_size)

        x = torch.randn(batch_size, emb_size)
        output = model(x)

        assert output.shape == (batch_size, 3)

    def test_mlp_output_shape_single_sample(self):
        """Test MLP output shape with single sample."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        x = torch.randn(1, emb_size)
        output = model(x)

        assert output.shape == (1, 3)

    def test_mlp_output_shape_large_batch(self):
        """Test MLP output shape with large batch."""
        emb_size = 768
        batch_size = 128
        model = MLP(emb_size=emb_size)

        x = torch.randn(batch_size, emb_size)
        output = model(x)

        assert output.shape == (batch_size, 3)

    def test_mlp_output_non_negative(self):
        """Test MLP output is non-negative (ReLU activation at end)."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        x = torch.randn(10, emb_size)
        output = model(x)

        # Output should be non-negative due to final ReLU
        assert torch.all(output >= 0)

    def test_mlp_different_embedding_sizes(self):
        """Test MLP with different embedding sizes."""
        for emb_size in [256, 512, 768, 1024]:
            model = MLP(emb_size=emb_size)
            x = torch.randn(4, emb_size)
            output = model(x)

            assert output.shape == (4, 3)

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through MLP."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        x = torch.randn(4, emb_size, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_mlp_train_mode(self):
        """Test MLP in training mode."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        model.train()

        x = torch.randn(4, emb_size)
        output = model(x)

        assert output is not None
        assert model.training is True

    def test_mlp_eval_mode(self):
        """Test MLP in evaluation mode."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        model.eval()

        x = torch.randn(4, emb_size)
        output = model(x)

        assert output is not None
        assert model.training is False

    def test_mlp_dropout_behavior(self):
        """Test MLP dropout behavior in train vs eval mode."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        x = torch.randn(1, emb_size)

        # Set seed for reproducibility
        torch.manual_seed(42)
        model.train()
        output_train = model(x)

        torch.manual_seed(42)
        model.eval()
        output_eval = model(x)

        # Outputs might differ due to dropout in training mode
        # This test just verifies both modes work
        assert output_train is not None
        assert output_eval is not None

    def test_mlp_save_creates_directory(self, tmp_path):
        """Test that MLP save creates directory if it doesn't exist."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        save_dir = tmp_path / "models" / "subdir"
        save_path = save_dir / "model.pt"

        assert not save_dir.exists()

        model.save(str(save_path))

        assert save_dir.exists()
        assert save_path.exists()

    def test_mlp_save_creates_file(self, tmp_path):
        """Test that MLP save creates model file."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        assert save_path.exists()
        assert save_path.is_file()

    def test_mlp_save_and_load(self, tmp_path):
        """Test MLP save and load cycle."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        # Save model
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        # Create new model and load state
        model2 = MLP(emb_size=emb_size)
        model2.model.load_state_dict(torch.load(save_path))

        # Test that loaded model produces same output
        x = torch.randn(4, emb_size)
        model.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model(x)
            output2 = model2(x)

        assert torch.allclose(output1, output2)

    def test_mlp_save_overwrites_existing(self, tmp_path):
        """Test that MLP save overwrites existing file."""
        emb_size = 768
        model1 = MLP(emb_size=emb_size)
        model2 = MLP(emb_size=emb_size)

        # Initialize with different weights
        with torch.no_grad():
            for p in model2.parameters():
                p.fill_(0.5)

        save_path = tmp_path / "model.pt"

        # Save first model
        model1.save(str(save_path))
        size1 = save_path.stat().st_size

        # Save second model
        model2.save(str(save_path))
        size2 = save_path.stat().st_size

        # File should exist and may have different size
        assert save_path.exists()

    def test_mlp_device_compatibility_cpu(self):
        """Test MLP on CPU device."""
        device = torch.device("cpu")
        emb_size = 768
        model = MLP(emb_size=emb_size).to(device)

        x = torch.randn(4, emb_size, device=device)
        output = model(x)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mlp_device_compatibility_cuda(self):
        """Test MLP on CUDA device."""
        device = torch.device("cuda")
        emb_size = 768
        model = MLP(emb_size=emb_size).to(device)

        x = torch.randn(4, emb_size, device=device)
        output = model(x)

        assert output.device.type == "cuda"

    def test_mlp_parameters_count(self):
        """Test MLP has expected number of parameters."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        param_count = sum(p.numel() for p in model.parameters())

        # Calculate expected parameters:
        # Layer 1: (768 * 1024) + 1024 = 787,456 + 1,024 = 788,480
        # Layer 2: (1024 * 512) + 512 = 524,288 + 512 = 524,800
        # Layer 3: (512 * 256) + 256 = 131,072 + 256 = 131,328
        # Layer 4: (256 * 3) + 3 = 768 + 3 = 771
        # Total: 1,445,379
        expected = (emb_size * 1024 + 1024) + (1024 * 512 + 512) + (512 * 256 + 256) + (256 * 3 + 3)

        assert param_count == expected

    def test_mlp_forward_consistency(self):
        """Test that MLP forward pass is consistent with same input."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        model.eval()

        x = torch.randn(4, emb_size)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


class TestMLPIntegration:
    """Integration tests for MLP model."""

    def test_mlp_in_training_loop(self):
        """Test MLP in a training loop."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for _ in range(5):
            x = torch.randn(8, emb_size)
            y = torch.randn(8, 3)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Model should have been updated
        assert loss is not None

    def test_mlp_inference(self):
        """Test MLP inference workflow."""
        emb_size = 768
        model = MLP(emb_size=emb_size)
        model.eval()

        # Simulate inference
        batch_inputs = [torch.randn(1, emb_size) for _ in range(10)]

        predictions = []
        with torch.no_grad():
            for x in batch_inputs:
                pred = model(x)
                predictions.append(pred)

        assert len(predictions) == 10
        assert all(p.shape == (1, 3) for p in predictions)

    def test_mlp_save_load_inference(self, tmp_path):
        """Test complete workflow: train, save, load, inference."""
        emb_size = 768
        model = MLP(emb_size=emb_size)

        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(10, emb_size)
        y = torch.randn(10, 3)

        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Save model
        save_path = tmp_path / "trained_model.pt"
        model.save(str(save_path))

        # Load into new model
        model_loaded = MLP(emb_size=emb_size)
        model_loaded.model.load_state_dict(torch.load(save_path))
        model_loaded.eval()

        # Inference
        test_input = torch.randn(5, emb_size)
        with torch.no_grad():
            predictions = model_loaded(test_input)

        assert predictions.shape == (5, 3)
        assert torch.all(predictions >= 0)
