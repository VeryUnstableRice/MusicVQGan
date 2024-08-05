import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.convert = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.convert(x) + self.block(x)


class MusicVQGan(nn.Module):
    def __init__(self, multipler):
        super(MusicVQGan, self).__init__()

        encoder_channels = [16, 8, 4, 2]
        encoder_layers = []
        channels = 1
        for encoder in encoder_channels:
            encoder_layers.append(ResBlock(channels, encoder * multipler))
            encoder_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            channels = encoder * multipler
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_channels = [2, 4, 8, 16]
        decoder_layers = []
        for decoder in decoder_channels:
            decoder_layers.append(ResBlock(channels, decoder * multipler))
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            channels = decoder * multipler
        encoder_layers.append(nn.Conv1d(channels, 1, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


model = MusicVQGan(32).cuda()
x = torch.randn(1, 1, 960000).cuda()
x = model(x)
print(x.shape)
