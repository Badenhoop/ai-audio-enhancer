import torch
import numpy as np
from pydub import AudioSegment
import librosa
import random
from glob import glob
import os


def load_audio(path):
    audio = AudioSegment.from_file(path)
    assert audio.frame_rate == 44100, f'Sample rate must be 44100 but got {audio.frame_rate} from "{path}".'
    audio = np.array(audio.get_array_of_samples(), dtype=np.float32)
    audio = librosa.util.normalize(audio)
    return audio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super(AudioDataset, self).__init__()
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        return load_audio(path)


def collate(minibatch, audio_length):
    result = []
    for audio in minibatch:
        n = audio.shape[-1]
        if n < audio_length:
            continue    
        start = random.randint(0, n - audio_length)
        end = start + audio_length
        result.append(audio[start:end])
    return torch.from_numpy(np.stack(result))


def build_dataloader(directory,
                     audio_format,
                     batch_size,
                     audio_length,
                     shuffle):
    paths = glob(f'{directory}/*.{audio_format}')
    dataset = AudioDataset(paths)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda minibatch: collate(minibatch, audio_length),
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True)