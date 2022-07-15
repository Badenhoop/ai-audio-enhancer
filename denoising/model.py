import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_diffusion_steps, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer('encoding', self._build_encoding(num_diffusion_steps), persistent=False)
        self.projection1 = nn.Linear(num_channels, num_channels)
        self.projection2 = nn.Linear(num_channels, num_channels)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.encoding[diffusion_step]
        else:
            x = self._lerp_encoding(diffusion_step)
        x = F.leaky_relu(self.projection1(x)) + x
        x = F.leaky_relu(self.projection2(x)) + x
        return x

    def _lerp_encoding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.encoding[low_idx]
        high = self.encoding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_encoding(self, num_diffusion_steps):
        steps = torch.arange(num_diffusion_steps).unsqueeze(1)
        dims = torch.arange(self.num_channels // 2).unsqueeze(0)
        encoding = 2. * np.pi * steps * np.exp(-np.log(2) * dims)
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=1)
        return encoding


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.LeakyReLU(inplace=True))
    
    def forward(self, x):
        return self.conv(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + out_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.LeakyReLU(inplace=True))
    
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2], mode='linear')
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class DiffusionUNetModel(nn.Module):
    def __init__(self, num_diffusion_steps):
        super(DiffusionUNetModel, self).__init__()
        self.diffusion_embedding = DiffusionEmbedding(
            num_diffusion_steps=num_diffusion_steps,
            num_channels=32)
        self.in_proj = nn.Conv1d(1, 64, 1)
        self.out_proj = nn.Conv1d(64, 1, 1)
        self.down = nn.ModuleList([
            DownsamplingBlock(64, 64),
            DownsamplingBlock(64, 64),
            DownsamplingBlock(64, 64),
            DownsamplingBlock(64, 64),
            DownsamplingBlock(64, 128),
            DownsamplingBlock(128, 128),
            DownsamplingBlock(128, 128),
            DownsamplingBlock(128, 128),
            DownsamplingBlock(128, 256),
            DownsamplingBlock(256, 256),
            DownsamplingBlock(256, 256),
            DownsamplingBlock(256, 256),
        ])
        self.up = nn.ModuleList([
            UpsamplingBlock(256, 256),
            UpsamplingBlock(256, 256),
            UpsamplingBlock(256, 256),
            UpsamplingBlock(256, 128),
            UpsamplingBlock(128, 128),
            UpsamplingBlock(128, 128),
            UpsamplingBlock(128, 128),
            UpsamplingBlock(128, 64),
            UpsamplingBlock(64, 64),
            UpsamplingBlock(64, 64),
            UpsamplingBlock(64, 64),
            UpsamplingBlock(64, 64),
        ])
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, audio, diffusion_step):
        embedding = self.diffusion_embedding(diffusion_step)
        x = audio.unsqueeze(1) # channel dimension
        x = self.in_proj(x)

        skip_connections = []
        for layer in self.down:
            skip_connections.append(x)
            N, C, T = x.shape
            embedding_padded = torch.zeros((N, C), dtype=x.dtype, device=x.device)
            embedding_padded[:, :embedding.shape[1]] = embedding
            x = x + embedding_padded[:, :, None]
            x = layer(x)

        for i, layer in enumerate(self.up):
            skip = skip_connections[len(skip_connections) - i - 1]
            x = layer(x, skip)

        x = self.out_proj(x)
        x = x.squeeze(1)
        return x