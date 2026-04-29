import torch
import torch.nn as nn
import torch.nn.functional as F

# Basé sur Shi et al. (2015)
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,  # i, f, o, g
            kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(4 * hidden_channels)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1) # concaténer puis convolutionner revient à faire les conv séparément et concat après
        gates = self.bn(self.conv(combined))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i) # input gate
        f = torch.sigmoid(f) # forget gate
        o = torch.sigmoid(o) # output gate
        g = torch.tanh(g) # cell gate
        c_next = f * c + i * g # état de cellule
        h_next = o * torch.tanh(c_next) # état caché
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


# Bloc de convolution classique pour le décodeur UNet : 2 convolutions 3x3 + BatchNorm + ReLU (padding de 1 pour garder la dimension)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

# Module de contexte global inspiré du global context block de CGNet de Wu et al. (2019) à partir de Kapp-Schwoerer et al. (2020).
class GlobalContextModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        gap = self.gap(x).view(b, c)
        out = self.sigmoid(self.fc2(torch.relu(self.fc1(gap))))
        return x * out.view(b, c, 1, 1)


# Encodeur ConvLSTM + Décodeur UNet
class ConvLSTMUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[8, 16, 32, 64]):
        super().__init__()
        self.features = features
        self.pool = nn.MaxPool2d(2, 2)

        # Encodeur ConvLSTM (Shi et al. (2015))
        self.lstm1 = ConvLSTMCell(in_channels, features[0])
        self.lstm2 = ConvLSTMCell(features[0], features[1]) # reçoit hidden states du niveau précédent
        self.lstm3 = ConvLSTMCell(features[1], features[2])
        self.lstm4 = ConvLSTMCell(features[2], features[3])

        # Décodeur UNet (Ronneberger et al. (2015))
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = ConvBlock(features[3], features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = ConvBlock(features[2], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = ConvBlock(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def encode_sequence(self, lstm_cell, x_seq):
        # passe la séquence dans la cellule ConvLSTM pour obtenir le dernier hidden state
        B, T, C, H, W = x_seq.shape
        h, c = lstm_cell.init_hidden(B, H, W, x_seq.device)
        for t in range(T):
            h, c = lstm_cell(x_seq[:, t], h, c)
        return h # (B, hidden_channels, H, W)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encodeur
        # Niveau 1 
        e1 = self.encode_sequence(self.lstm1, x) # e1 : (B, features[0], H, W)

        # Niveau 2 :
        # On pool e1 pour chaque timestep en partant des features encodés
        e1_seq = e1.unsqueeze(1).expand(-1, T, -1, -1, -1)  # ajoute dimension T : (B, T, f0, H, W)
        e1_pooled = torch.stack([self.pool(e1_seq[:, t]) for t in range(T)], dim=1) # pool chaque frame : (B, T, f0, H/2, W/2)
        e2 = self.encode_sequence(self.lstm2, e1_pooled) # e2 : (B, features[1], H/2, W/2)

        # Niveau 3 : hidden states niveau 2 poolés
        e2_seq = e2.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e2_pooled = torch.stack([self.pool(e2_seq[:, t]) for t in range(T)], dim=1) # (B, T, f1, H/4, W/4)
        e3 = self.encode_sequence(self.lstm3, e2_pooled) # e3 : (B, features[2], H/4, W/4)

        # Niveau 4 (bottleneck) : hidden states niveau 3 poolés
        e3_seq = e3.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e3_pooled = torch.stack([self.pool(e3_seq[:, t]) for t in range(T)], dim=1)  # (B, T, f2, H/8, W/8)
        e4 = self.encode_sequence(self.lstm4, e3_pooled) # e4 : (B, features[3], H/8, W/8)

        # Décodeur
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1)) # skip connection

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
    
    
# Variante du ConvLSTMUNet avec un module de contexte global sur le goulot d'étranglement, inspiré du global context block de CGNet
class ConvLSTMUNetGC(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[8, 16, 32, 64], dropout=0.0):
        super().__init__()
        self.features = features
        self.pool = nn.MaxPool2d(2, 2)

        self.use_dropout = dropout > 0  # Désactivé par défaut
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout)

        # Encodeur ConvLSTM (identique)
        self.lstm1 = ConvLSTMCell(in_channels,  features[0])
        self.lstm2 = ConvLSTMCell(features[0],  features[1])
        self.lstm3 = ConvLSTMCell(features[1],  features[2])
        self.lstm4 = ConvLSTMCell(features[2],  features[3])

        # Global Context sur le goulot d'étranglement
        self.global_context = GlobalContextModule(features[3])

        # Décodeur UNet (identique)
        self.up3  = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = ConvBlock(features[3], features[2])

        self.up2  = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = ConvBlock(features[2], features[1])

        self.up1  = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = ConvBlock(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def encode_sequence(self, lstm_cell, x_seq):
        B, T, C, H, W = x_seq.shape
        h, c = lstm_cell.init_hidden(B, H, W, x_seq.device)
        for t in range(T):
            h, c = lstm_cell(x_seq[:, t], h, c)
        return h

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encodeur
        e1 = self.encode_sequence(self.lstm1, x)
        if self.use_dropout:
            e1 = self.dropout(e1)

        e1_seq = e1.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e1_pooled = torch.stack([self.pool(e1_seq[:, t]) for t in range(T)], dim=1)
        e2 = self.encode_sequence(self.lstm2, e1_pooled)
        if self.use_dropout:
            e2 = self.dropout(e2)

        e2_seq = e2.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e2_pooled = torch.stack([self.pool(e2_seq[:, t]) for t in range(T)], dim=1)
        e3 = self.encode_sequence(self.lstm3, e2_pooled)
        if self.use_dropout:
            e3 = self.dropout(e3)

        e3_seq = e3.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e3_pooled = torch.stack([self.pool(e3_seq[:, t]) for t in range(T)], dim=1)
        e4 = self.encode_sequence(self.lstm4, e3_pooled)
        if self.use_dropout:
            e4 = self.dropout(e4)

        # Global Context
        e4 = self.global_context(e4)

        # Décodeur
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
    

class ConvLSTMUNetFullLSTM(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[8, 16, 32, 64]):
        super().__init__()
        self.features = features
        self.pool = nn.MaxPool2d(2, 2)

        # Encodeur ConvLSTM (identique)
        self.lstm1 = ConvLSTMCell(in_channels,  features[0])
        self.lstm2 = ConvLSTMCell(features[0],  features[1])
        self.lstm3 = ConvLSTMCell(features[1],  features[2])
        self.lstm4 = ConvLSTMCell(features[2],  features[3])

        # Décodeur ConvLSTM
        # Chaque niveau reçoit : upsample(e_below) + skip connection
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dlstm3 = ConvLSTMCell(features[3], features[2])  # features[3] car concat skip

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dlstm2 = ConvLSTMCell(features[2], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dlstm1 = ConvLSTMCell(features[1], features[0])

        self.out_conv = nn.Conv2d(features[0], out_channels, 1)

    def encode_sequence(self, lstm_cell, x_seq):
        B, T, C, H, W = x_seq.shape
        h, c = lstm_cell.init_hidden(B, H, W, x_seq.device)
        for t in range(T):
            h, c = lstm_cell(x_seq[:, t], h, c)
        return h

    def decode_step(self, lstm_cell, x, h, c):
        """Un seul pas de décodage ConvLSTM."""
        return lstm_cell(x, h, c)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Encodeur (identique)
        e1 = self.encode_sequence(self.lstm1, x)

        e1_seq   = e1.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e1_pooled = torch.stack([self.pool(e1_seq[:, t]) for t in range(T)], dim=1)
        e2 = self.encode_sequence(self.lstm2, e1_pooled)

        e2_seq    = e2.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e2_pooled = torch.stack([self.pool(e2_seq[:, t]) for t in range(T)], dim=1)
        e3 = self.encode_sequence(self.lstm3, e2_pooled)

        e3_seq    = e3.unsqueeze(1).expand(-1, T, -1, -1, -1)
        e3_pooled = torch.stack([self.pool(e3_seq[:, t]) for t in range(T)], dim=1)
        e4 = self.encode_sequence(self.lstm4, e3_pooled)

        # Décodeur ConvLSTM
        # Niveau 3
        d3 = self.up3(e4)
        d3_cat = torch.cat([d3, e3], dim=1)  # (B, features[3], H/4, W/4) skip connections
        h3, c3 = self.dlstm3.init_hidden(B, d3_cat.shape[2], d3_cat.shape[3], x.device)
        # différence avec ConvLSTMUNet : au lieu d'un ConvBlock -> cellule ConvLSTM
        d3_out, _ = self.decode_step(self.dlstm3, d3_cat, h3, c3)

        # Niveau 2
        d2 = self.up2(d3_out)
        d2_cat = torch.cat([d2, e2], dim=1)
        h2, c2 = self.dlstm2.init_hidden(B, d2_cat.shape[2], d2_cat.shape[3], x.device)
        d2_out, _ = self.decode_step(self.dlstm2, d2_cat, h2, c2)

        # Niveau 1
        d1 = self.up1(d2_out)
        d1_cat = torch.cat([d1, e1], dim=1)
        h1, c1 = self.dlstm1.init_hidden(B, d1_cat.shape[2], d1_cat.shape[3], x.device)
        d1_out, _ = self.decode_step(self.dlstm1, d1_cat, h1, c1)

        return self.out_conv(d1_out)
    

# ConvLSTM simple à résolution pleine (sans pooling), sans décodeur UNet
class ConvLSTMPure(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, hidden_channels=[8, 16, 8]):
        super().__init__()

        self.lstm1 = ConvLSTMCell(in_channels, hidden_channels[0]) # 4 -> 8
        self.lstm2 = ConvLSTMCell(hidden_channels[0], hidden_channels[1]) # 8 -> 16
        self.lstm3 = ConvLSTMCell(hidden_channels[1], hidden_channels[2]) # 16 -> 8
        self.out_conv = nn.Conv2d(hidden_channels[2], out_channels, 1) # 8 -> 3 classes

    def forward(self, x):
        B, T, C, H, W = x.shape

        h1, c1 = self.lstm1.init_hidden(B, H, W, x.device) # résolution HxW
        h2, c2 = self.lstm2.init_hidden(B, H, W, x.device)
        h3, c3 = self.lstm3.init_hidden(B, H, W, x.device)
        for t in range(T):
            h1, c1 = self.lstm1(x[:, t], h1, c1) # hidden state devient entrée du niveau suivant
            h2, c2 = self.lstm2(h1, h2, c2)
            h3, c3 = self.lstm3(h2, h3, c3)

        return self.out_conv(h3)
