import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import os


class SongDataset(Dataset):
    def __init__(self, training_dir, segment_lenght=48000):
        super().__init__()
        self.filenames = [f for f in os.listdir(training_dir) if f.endswith('.mp3')]
        self.training_dir = training_dir
        self.segment_lenght = segment_lenght

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        file_path = os.path.join(self.training_dir, self.filenames[item])
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] > self.segment_lenght:
            max = waveform.shape[1] - self.segment_lenght
            start = random.randint(0, max)
            waveform = waveform[:, start:start+self.segment_lenght]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_lenght-waveform.shape[1]))

        return waveform


def save_audio(tensor, output_dir, file_name):
    torchaudio.save(os.path.join(output_dir, file_name), tensor, 16000)