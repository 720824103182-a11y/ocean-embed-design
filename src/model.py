import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    Stage 7 & 8: CNN Encoder for Spatial Feature Extraction
    Input: [Batch, C_in, H, W] (5 surface ocean channels)
    Output: [Batch, F, H, W] (Learned Ocean Representation)
    """
    def __init__(self, in_channels=5, feature_dim=64):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=feature_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        features = self.relu2(self.conv2(h))
        return features


class PointwiseDNN(nn.Module):
    """
    Stage 9: Shared Feed-Forward DNN across grid locations
    Input: [Batch, F, H, W] (Learned Ocean Representation)
    Output: [Batch, out_depths, H, W] (15 predicted subsurface temperature levels)
    """
    def __init__(self, feature_dim=64, hidden_dim=128, out_depths=15):
        super(PointwiseDNN, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=out_depths, kernel_size=1)

    def forward(self, features):
        h1 = self.relu1(self.fc1(features))
        h2 = self.relu2(self.fc2(h1))
        out = self.fc3(h2)
        return out


class OceanEmbedModel(nn.Module):
    """
    Stage 10: Complete OceanEmbed Framework (CNN Encoder + Pointwise DNN)
    Connects 5 surface observations [B, 5, H, W] -> 15 Subsurface Temperature Fields [B, 15, H, W]
    """
    def __init__(self, in_channels=5, feature_dim=64, hidden_dim=128, out_depths=15):
        super(OceanEmbedModel, self).__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, feature_dim=feature_dim)
        self.decoder = PointwiseDNN(feature_dim=feature_dim, hidden_dim=hidden_dim, out_depths=out_depths)

    def forward(self, x):
        feat = self.encoder(x)
        pred = self.decoder(feat)
        return pred

if __name__ == "__main__":
    dummy_input = torch.randn(2, 5, 101, 241)
    model = OceanEmbedModel(in_channels=5, feature_dim=64, hidden_dim=128, out_depths=15)
    output = model(dummy_input)
    print("Model test successful!")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape} (15 Subsurface Temperature Depths)")
