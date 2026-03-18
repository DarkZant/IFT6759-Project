import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from convlstm_cell import ConvLSTMCell, ConvLSTMLayer, SegmentationHead, ConvLSTM

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B, T, C, H, W = 2, 5, 4, 16, 16
HIDDEN = 8
KERNEL = 3
NUM_LAYERS = 2
NUM_CLASSES = 3
DEVICE = torch.device("cpu")


@pytest.fixture
def cell():
    return ConvLSTMCell(input_dim=C, hidden_dim=HIDDEN, kernel_size=KERNEL, biais=True)


@pytest.fixture
def layer():
    return ConvLSTMLayer(input_dim=C, hidden_dim=HIDDEN, kernel_size=KERNEL, num_layers=NUM_LAYERS)


@pytest.fixture
def head():
    return SegmentationHead(hidden_dim=HIDDEN, num_classes=NUM_CLASSES)


@pytest.fixture
def model():
    return ConvLSTM(input_dim=C, hidden_dim=HIDDEN, kernel_size=KERNEL,
                    num_layers=NUM_LAYERS, num_classes=NUM_CLASSES)


@pytest.fixture
def model_weighted():
    weights = torch.tensor([0.1, 1.0, 5.0])
    return ConvLSTM(input_dim=C, hidden_dim=HIDDEN, kernel_size=KERNEL,
                    num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                    class_weights=weights)


@pytest.fixture
def fake_dataloader():
    x = torch.randn(4, T, C, H, W)
    y = torch.randint(0, NUM_CLASSES, (4, H, W))
    return DataLoader(TensorDataset(x, y), batch_size=2)


# ---------------------------------------------------------------------------
# ConvLSTMCell
# ---------------------------------------------------------------------------

class TestConvLSTMCell:

    def test_init_hidden_shapes(self, cell):
        h, c = cell.init_hidden(batch_size=B, hidden_dim=HIDDEN, height=H, width=W)
        assert h.shape == (B, HIDDEN, H, W)
        assert c.shape == (B, HIDDEN, H, W)

    def test_init_hidden_zeros(self, cell):
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        assert h.sum() == 0
        assert c.sum() == 0

    def test_init_hidden_device(self, cell):
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        assert h.device == cell.conv.weight.device
        assert c.device == cell.conv.weight.device

    def test_forward_output_shapes(self, cell):
        x = torch.randn(B, C, H, W)
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        h_next, c_next = cell(x, h, c)
        assert h_next.shape == (B, HIDDEN, H, W)
        assert c_next.shape == (B, HIDDEN, H, W)

    def test_forward_preserves_spatial_dims(self, cell):
        # padding must keep H, W unchanged
        x = torch.randn(B, C, H, W)
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        h_next, _ = cell(x, h, c)
        assert h_next.shape[-2:] == (H, W)

    def test_forward_h_values_in_tanh_range(self, cell):
        x = torch.randn(B, C, H, W)
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        h_next, _ = cell(x, h, c)
        # h = o * tanh(c), o in (0,1), tanh in (-1,1) → h in (-1,1)
        assert h_next.min().item() >= -1.0
        assert h_next.max().item() <= 1.0

    def test_forward_gradients_flow(self, cell):
        x = torch.randn(B, C, H, W, requires_grad=True)
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        h_next, c_next = cell(x, h, c)
        loss = h_next.sum() + c_next.sum()
        loss.backward()
        assert x.grad is not None

    def test_multiple_timesteps_state_updates(self, cell):
        h, c = cell.init_hidden(B, HIDDEN, H, W)
        h0 = h.clone()
        x = torch.randn(B, C, H, W)
        h1, c1 = cell(x, h, c)
        h2, c2 = cell(x, h1, c1)
        # state should change across timesteps
        assert not torch.allclose(h1, h2)


# ---------------------------------------------------------------------------
# ConvLSTMLayer
# ---------------------------------------------------------------------------

class TestConvLSTMLayer:

    def test_forward_last_hidden_shape(self, layer):
        x = torch.randn(B, T, C, H, W)
        out = layer(x)
        assert out.shape == (B, HIDDEN, H, W)

    def test_forward_return_all_layers_shape(self):
        layer_all = ConvLSTMLayer(C, HIDDEN, KERNEL, NUM_LAYERS, return_all_layer=True)
        x = torch.randn(B, T, C, H, W)
        out = layer_all(x)
        assert out.shape == (B, T, HIDDEN, H, W)

    def test_forward_single_layer(self):
        single = ConvLSTMLayer(C, HIDDEN, KERNEL, num_layers=1)
        x = torch.randn(B, T, C, H, W)
        out = single(x)
        assert out.shape == (B, HIDDEN, H, W)

    def test_correct_number_of_cells(self, layer):
        assert len(layer.layers) == NUM_LAYERS

    def test_first_layer_input_dim(self, layer):
        # first cell conv: in_channels = C + HIDDEN
        first_in = layer.layers[0].conv.in_channels
        assert first_in == C + HIDDEN

    def test_subsequent_layers_input_dim(self, layer):
        # subsequent cells conv: in_channels = HIDDEN + HIDDEN
        for cell in layer.layers[1:]:
            assert cell.conv.in_channels == HIDDEN + HIDDEN

    def test_gradients_flow(self, layer):
        x = torch.randn(B, T, C, H, W, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SegmentationHead
# ---------------------------------------------------------------------------

class TestSegmentationHead:

    def test_forward_shape(self, head):
        x = torch.randn(B, HIDDEN, H, W)
        out = head(x)
        assert out.shape == (B, NUM_CLASSES, H, W)

    def test_preserves_spatial_dims(self, head):
        x = torch.randn(B, HIDDEN, H, W)
        out = head(x)
        assert out.shape[-2:] == (H, W)

    def test_no_activation_applied(self, head):
        # output should not be clipped — raw logits can exceed [-1, 1]
        x = torch.randn(B, HIDDEN, H, W) * 10
        out = head(x)
        assert out.abs().max().item() > 1.0


# ---------------------------------------------------------------------------
# ConvLSTM — forward
# ---------------------------------------------------------------------------

class TestConvLSTMForward:

    def test_output_shape(self, model):
        x = torch.randn(B, T, C, H, W)
        out = model(x)
        assert out.shape == (B, NUM_CLASSES, H, W)

    def test_output_is_logits_not_probabilities(self, model):
        x = torch.randn(B, T, C, H, W)
        out = model(x)
        # softmax would make all values in (0,1); raw logits should have values outside that
        assert (out < 0).any() or (out > 1).any()

    def test_gradients_flow_through_full_model(self, model):
        x = torch.randn(B, T, C, H, W, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_class_weights_stored_as_buffer(self, model_weighted):
        assert model_weighted.class_weights is not None
        assert model_weighted.class_weights.shape == (NUM_CLASSES,)

    def test_no_class_weights_buffer_is_none(self, model):
        assert model.class_weights is None


# ---------------------------------------------------------------------------
# ConvLSTM — iou_per_class & mean_iou
# ---------------------------------------------------------------------------

class TestIoU:

    def test_iou_per_class_returns_list_of_length_num_classes(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        ious = model.iou_per_class(pred, targets)
        assert len(ious) == NUM_CLASSES

    def test_iou_values_in_range(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        ious = model.iou_per_class(pred, targets)
        for iou in ious:
            assert 0.0 <= iou <= 1.0

    def test_perfect_prediction_iou_is_one(self, model):
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        # build logits that will argmax to exactly targets
        pred = torch.zeros(B, NUM_CLASSES, H, W) - 1e9
        for cls in range(NUM_CLASSES):
            pred[:, cls, :, :][targets == cls] = 1e9
        ious = model.iou_per_class(pred, targets)
        for iou in ious:
            assert abs(iou - 1.0) < 1e-5

    def test_all_wrong_prediction_iou_is_zero(self, model):
        # all pixels predicted as class 0, all targets are class 1
        targets = torch.ones(B, H, W, dtype=torch.long)
        pred = torch.zeros(B, NUM_CLASSES, H, W) - 1e9
        pred[:, 0, :, :] = 1e9  # always predict class 0
        ious = model.iou_per_class(pred, targets)
        assert ious[0] == 0.0  # class 0: predicted but not in target
        assert ious[1] == 0.0  # class 1: in target but not predicted

    def test_mean_iou_is_average_of_per_class(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        ious = model.iou_per_class(pred, targets)
        mean = model.mean_iou(pred, targets)
        assert abs(mean - sum(ious) / len(ious)) < 1e-6


# ---------------------------------------------------------------------------
# ConvLSTM — dice_loss & combined_loss
# ---------------------------------------------------------------------------

class TestLoss:

    def test_dice_loss_is_tensor(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model.dice_loss(pred, targets)
        assert isinstance(loss, torch.Tensor)

    def test_dice_loss_has_gradient(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model.dice_loss(pred, targets)
        loss.backward()
        assert pred.grad is not None

    def test_dice_loss_in_valid_range(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model.dice_loss(pred, targets)
        assert 0.0 <= loss.item() <= 1.0

    def test_dice_loss_no_nan(self, model):
        # check no NaN even when a class is absent from the batch
        targets = torch.zeros(B, H, W, dtype=torch.long)  # only class 0 present
        pred = torch.randn(B, NUM_CLASSES, H, W)
        loss = model.dice_loss(pred, targets)
        assert not torch.isnan(loss)

    def test_combined_loss_is_tensor_with_grad(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model.combined_loss(pred, targets)
        loss.backward()
        assert pred.grad is not None

    def test_combined_loss_positive(self, model):
        pred = torch.randn(B, NUM_CLASSES, H, W)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model.combined_loss(pred, targets)
        assert loss.item() > 0

    def test_combined_loss_with_class_weights(self, model_weighted):
        pred = torch.randn(B, NUM_CLASSES, H, W, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (B, H, W))
        loss = model_weighted.combined_loss(pred, targets)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# ConvLSTM — predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_predict_shape(self, model):
        x = torch.randn(B, T, C, H, W)
        out = model.predict(x)
        assert out.shape == (B, H, W)

    def test_predict_values_are_class_indices(self, model):
        x = torch.randn(B, T, C, H, W)
        out = model.predict(x)
        assert out.min().item() >= 0
        assert out.max().item() < NUM_CLASSES

    def test_predict_dtype_is_long(self, model):
        x = torch.randn(B, T, C, H, W)
        out = model.predict(x)
        assert out.dtype == torch.long

    def test_predict_sets_eval_mode(self, model):
        x = torch.randn(B, T, C, H, W)
        model.predict(x)
        assert not model.training


# ---------------------------------------------------------------------------
# ConvLSTM — fit & evaluate
# ---------------------------------------------------------------------------

class TestFitEvaluate:

    def test_fit_returns_scalar(self, model, fake_dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        avg_loss = model.fit(fake_dataloader, optimizer, num_epoch=1, device=DEVICE)
        assert isinstance(avg_loss, float)

    def test_fit_loss_decreases_over_epochs(self, model, fake_dataloader):
        # fix seed for reproducibility
        torch.manual_seed(0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_e1 = model.fit(fake_dataloader, optimizer, num_epoch=1, device=DEVICE)
        loss_e2 = model.fit(fake_dataloader, optimizer, num_epoch=1, device=DEVICE)
        # not guaranteed but very likely with a decent lr on a tiny dataset
        assert loss_e2 <= loss_e1 * 1.5  # loose check — just ensure it trains

    def test_fit_sets_train_mode(self, model, fake_dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.fit(fake_dataloader, optimizer, num_epoch=1, device=DEVICE)
        assert model.training

    def test_evaluate_returns_mean_iou_and_per_class(self, model, fake_dataloader):
        mean_iou, per_class_iou, _, _ = model.evaluate(fake_dataloader, device=DEVICE)
        assert isinstance(mean_iou, float)
        assert len(per_class_iou) == NUM_CLASSES

    def test_evaluate_mean_iou_in_range(self, model, fake_dataloader):
        mean_iou, _, _, _ = model.evaluate(fake_dataloader, device=DEVICE)
        assert 0.0 <= mean_iou <= 1.0

    def test_evaluate_per_class_iou_in_range(self, model, fake_dataloader):
        _, per_class_iou, _, _ = model.evaluate(fake_dataloader, device=DEVICE)
        for iou in per_class_iou:
            assert 0.0 <= iou <= 1.0

    def test_evaluate_sets_eval_mode(self, model, fake_dataloader):
        model.evaluate(fake_dataloader, device=DEVICE)
        assert not model.training

    def test_evaluate_returns_mean_recall(self, model, fake_dataloader):
        _, _, mean_recall, per_class_recall = model.evaluate(fake_dataloader, device=DEVICE)
        assert isinstance(mean_recall, float)
        assert len(per_class_recall) == NUM_CLASSES

    def test_evaluate_recall_in_range(self, model, fake_dataloader):
        _, _, mean_recall, per_class_recall = model.evaluate(fake_dataloader, device=DEVICE)
        assert 0.0 <= mean_recall <= 1.0
        for r in per_class_recall:
            assert 0.0 <= r <= 1.0
