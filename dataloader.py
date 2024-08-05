import torchaudio
from torch.utils.data import Dataset, DataLoader
import os


class SongDataset(Dataset):
    def __init__(self, training_dir):
        super().__init__()
        self.filenames = [f for f in os.listdir(training_dir) if f.endswith('.mp3')]
        self.training_dir = training_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        file_path = os.path.join(self.training_dir, self.filenames[item])
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        return waveform


root_dir = './training_data'
dataset = SongDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i, data in enumerate(dataloader):
    print(data.shape)
    if i == 5:  # Limit the output for demonstration purposes
        break