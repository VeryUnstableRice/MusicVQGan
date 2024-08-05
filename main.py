import os

import dataloader
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from vector_quantize_pytorch import VectorQuantize

# https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
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


# https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
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
    def __init__(self, multiplier, bottleneck_size = 16):
        super(MusicVQGan, self).__init__()

        self.pixel_unshuffle = PixelUnshuffle1D(downscale_factor=4)

        encoder_channels = [32, 16, 8, 4, 2, 1]
        resnetnum_encoder = 2
        encoder_layers = []
        channels = 4
        for encoder in encoder_channels:
            for _ in range(resnetnum_encoder):
                encoder_layers.append(ResBlock(channels, encoder * multiplier))
                channels = encoder * multiplier
            encoder_layers.append(nn.AvgPool1d(kernel_size=2, stride=2))

        encoder_layers.append(nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_encoder = nn.Sequential(nn.Conv1d(channels, bottleneck_size, kernel_size=3, stride=1, padding=1),
                                                nn.LeakyReLU(),
                                                PixelUnshuffle1D(downscale_factor=5))

        self.vq = VectorQuantize(
            dim=bottleneck_size*5,
            codebook_size=256,
            use_cosine_sim=True
        )

        self.bottleneck_decoder = nn.Sequential(PixelShuffle1D(upscale_factor=5),
                                                nn.Conv1d(bottleneck_size, channels, kernel_size=3, stride=1, padding=1),
                                                nn.LeakyReLU())

        decoder_channels = [1, 2, 4, 8, 16, 32]
        decoder_layers = []
        resnetnum_decoder = 3
        for decoder in decoder_channels:
            for _ in range(resnetnum_decoder):
                decoder_layers.append(ResBlock(channels, decoder * multiplier))
                channels = decoder * multiplier
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        decoder_layers.append(nn.Conv1d(channels, 4, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.InstanceNorm1d(1))
        self.decoder = nn.Sequential(*decoder_layers)

        self.pixel_shuffle = PixelShuffle1D(upscale_factor=4)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.encoder(x)
        x = self.bottleneck_encoder(x)
        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vq(x)
        x = x.permute(0, 2, 1)
        x = self.bottleneck_decoder(x)
        x = self.decoder(x)
        x = self.pixel_shuffle(x)
        return x, commit_loss


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


model = MusicVQGan(32).cuda()
print_model_parameters(model)

root_dir = './training_data'
dataset = dataloader.SongDataset(root_dir)
dataloader_ = dataloader.DataLoader(dataset, batch_size=1, shuffle=True)

steps = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = GradScaler()

checkpoints_dir = './checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)

for epoch in range(128):
    with tqdm(dataloader_, desc=f'Epoch {epoch + 1}') as pbar:
        for i, data in enumerate(pbar):
            data_cuda = data.cuda()

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                h, vq_loss = model(data_cuda)
                loss_value = criterion(data_cuda, h) + vq_loss

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            if steps % 50 == 0:
                dataloader.save_audio(h[0].cpu().float(), './output', f'kebab_{steps}.mp3')
            steps += 1

            pbar.set_postfix(steps=steps, loss=f'{loss_value.item():.02f}')
    if epoch % 5 == 0:
        checkpoint_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')
