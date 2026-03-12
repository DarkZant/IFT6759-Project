import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, biais):
        super(ConvLSTMCell, self).__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=4 * hidden_dim, 
                  kernel_size=kernel_size, padding=self.padding, bias=biais)
        
    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        i, f, o, g = torch.chunk(self.conv(combined), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f*c_prev + i*g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next
    
    def init_hidden(self, batch_size, hidden_dim, height, width):
        h = torch.zeros(batch_size, hidden_dim, height, width, device=self.conv.weight.device)
        c = torch.zeros(batch_size, hidden_dim, height, width, device=self.conv.weight.device)
        return h, c
    

class ConvLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, return_all_layer=False):
        super(ConvLSTMLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.return_all_layer = return_all_layer
        self.hidden_dim = hidden_dim
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size, biais=True))

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        for layer in self.layers:
            h, c = layer.init_hidden(batch_size, self.hidden_dim, height, width)
            outputs = []
            for t in range(seq_len):
                h, c = layer(x[:, t], h, c)
                outputs.append(h)
            x = torch.stack(outputs, dim=1)
        return x if self.return_all_layer else x[:, -1]
    
class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out
    

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, num_classes=3, class_weights=None):
        super(ConvLSTM, self).__init__()
        self.num_classes = num_classes
        self.convlstm = ConvLSTMLayer(input_dim, hidden_dim, kernel_size, num_layers)
        self.segmentation_head = SegmentationHead(hidden_dim, self.num_classes)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

    def forward(self, x):
        x = self.convlstm(x)
        out = self.segmentation_head(x)
        return out
    
    def iou_per_class(self, pred, targets):
        pred_labels = torch.argmax(pred, dim=1)
        ious = []
        for cls in range(self.num_classes):
            pred_cls = (pred_labels == cls)
            target_cls = (targets == cls)
            intersection = (pred_cls & target_cls).sum().item()
            union = (pred_cls | target_cls).sum().item()
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        return ious

    def mean_iou(self, pred, targets):
        ious = self.iou_per_class(pred, targets)
        return sum(ious) / len(ious)

    def recall_per_class(self, pred, targets):
        pred_labels = torch.argmax(pred, dim=1)
        recalls = []
        for cls in range(self.num_classes):
            pred_cls    = (pred_labels == cls)
            target_cls  = (targets == cls)
            tp          = (pred_cls & target_cls).sum().item()
            actual      = target_cls.sum().item()
            recalls.append(tp / actual if actual > 0 else 0.0)
        return recalls

    def mean_recall(self, pred, targets):
        recalls = self.recall_per_class(pred, targets)
        return sum(recalls) / len(recalls)

    def dice_loss(self, pred, targets):
        probs = torch.softmax(pred, dim=1)
        dice_loss = 0
        for cls in range(self.num_classes):
            pred_cls = probs[:, cls]
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            dice_loss = dice_loss + (1 - (2 * intersection / (union + 1e-8)))
        return dice_loss / self.num_classes

    def combined_loss(self, pred, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)(pred, targets)
        dice = self.dice_loss(pred, targets)
        return ce_loss + dice

    def fit(self, dataloader, optimizer, num_epoch, device):
        self.train()
        total_loss = 0
        for epoch in range(num_epoch):
            epoch_loss = 0
            for x, targets in dataloader:
                x, targets = x.to(device), targets.to(device)
                optimizer.zero_grad()
                pred = self.forward(x)
                loss = self.combined_loss(pred, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epoch} — loss: {epoch_loss / len(dataloader):.4f}")
        return total_loss / (num_epoch * len(dataloader))

    def evaluate(self, dataloader, device):
        self.eval()
        total_iou    = 0
        total_recall = 0
        all_per_class_iou    = []
        all_per_class_recall = []
        with torch.no_grad():
            for x, targets in dataloader:
                x, targets = x.to(device), targets.to(device)
                pred = self.forward(x)
                total_iou    += self.mean_iou(pred, targets)
                total_recall += self.mean_recall(pred, targets)
                all_per_class_iou.append(self.iou_per_class(pred, targets))
                all_per_class_recall.append(self.recall_per_class(pred, targets))
        n = len(dataloader)
        mean_iou         = total_iou    / n
        mean_recall      = total_recall / n
        per_class_iou    = [sum(cls) / len(cls) for cls in zip(*all_per_class_iou)]
        per_class_recall = [sum(cls) / len(cls) for cls in zip(*all_per_class_recall)]
        return mean_iou, per_class_iou, mean_recall, per_class_recall
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
            pred_labels = torch.argmax(pred, dim=1)
        return pred_labels
        