import torch
import torch.nn as nn

#https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


#https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(out_channels)
        )
        self.convert = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.convert(x) + self.block(x)


class MusicVQGan(nn.Module):
    def __init__(self, multiplier):
        super(MusicVQGan, self).__init__()

        encoder_channels = [32, 16, 8, 4, 2, 1]
        encoder_layers = []
        channels = 4
        for encoder in encoder_channels:
            encoder_layers.append(ResBlock(channels, encoder * multiplier))
            encoder_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))
            channels = encoder * multiplier

        encoder_layers.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)
        self.bottleneck_encoder = nn.Sequential(nn.Conv1d(channels, 16, kernel_size=3, stride=1, padding=1),
                                                nn.LeakyReLU())
        self.bottleneck_decoder = nn.Sequential(nn.Conv1d(16, channels, kernel_size=3, stride=1, padding=1),
                                                nn.LeakyReLU())

        # Adding PixelUnshuffle1D before encoder
        self.pixel_unshuffle = PixelUnshuffle1D(downscale_factor=4)

        decoder_channels = [1, 2, 4, 8, 16, 32]
        decoder_layers = []
        for decoder in decoder_channels:
            decoder_layers.append(ResBlock(channels, decoder * multiplier))
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            channels = decoder * multiplier
        decoder_layers.append(nn.Conv1d(channels, 4, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.InstanceNorm1d(1))
        self.decoder = nn.Sequential(*decoder_layers)

        # Adding PixelShuffle1D after decoder
        self.pixel_shuffle = PixelShuffle1D(upscale_factor=4)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.encoder(x)
        x = self.bottleneck_encoder(x)
        x = self.bottleneck_decoder(x)
        x = self.decoder(x)
        x = self.pixel_shuffle(x)
        return x


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


model = MusicVQGan(32).cuda()
print_model_parameters(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss = nn.L1Loss()
for _ in range(64*256):
    x = torch.randn(1, 1, 480000).cuda()
    h = model(x)
    optimizer.zero_grad()
    loss_value = loss(x, h)
    loss_value.backward()
    optimizer.step()
    print(f'loss: {loss_value:.02f}')
