import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'climatenet'))
import torch
import torch.nn as nn
import torch.nn.functional as F

from climatenet.models import CGNetModule
from ConvLSTM.convlstm_cell import ConvLSTMLayer, SegmentationHead, ConvLSTM


class CGNetEncoder(CGNetModule):
    """
    CGNetModule with one extra method: encode().
    encode() runs the full 3-stage forward pass but stops before
    the classifier head, returning (B, 256, H/8, W/8) feature maps.

    We subclass instead of modifying the original file.
    """

    def encode(self, input):
        # Stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # Stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # Stage 3
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        return output2_cat  # (B, 256, H/8, W/8)


class CGNetConvLSTM(ConvLSTM):

    def __init__(
        self,
        hidden_dim,
        kernel_size,
        num_layers,
        num_classes=3,
        class_weights=None,
        cgnet_weights_path=None,
        freeze_encoder=True,
        channels=4,
    ):
        nn.Module.__init__(self)
        self.num_classes = num_classes

        # Spatial encoder (CGnet)
        self.encoder = CGNetEncoder(classes=num_classes, channels=channels)
        if cgnet_weights_path is not None:
            state = torch.load(cgnet_weights_path, map_location='cpu')
            self.encoder.load_state_dict(state)
            print(f"Loaded CGNet weights from {cgnet_weights_path}")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("CGNet encoder frozen.")

        self.convlstm = ConvLSTMLayer(
            input_dim=256,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )

        # Classifier head
        self.segmentation_head = SegmentationHead(hidden_dim, num_classes)

        # Class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

    def forward(self, x):

        B, T, C, H, W = x.shape

        # 1. Extract spatial features
        features = []
        for t in range(T):
            feat = self.encoder.encode(x[:, t]) 
            features.append(feat)
        features = torch.stack(features, dim=1)

        # 2. Temporal 
        last_hidden = self.convlstm(features)

        # 3. Classify
        logits = self.segmentation_head(last_hidden)

        # 4. Upsample back to original spatial resolution
        out = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return out  # (B, num_classes, H, W)
