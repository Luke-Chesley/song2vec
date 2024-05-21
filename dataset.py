import torch
from torch.utils.data import Dataset
import torchaudio
import os
import random
from dataclasses import dataclass
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, file_list, target_sample_rate=16000,max_length=80000):
        self.file_list = file_list
        self.target_sample_rate = target_sample_rate
        self.max_length = max_length
        self.label_list = sorted(self.labels()) 
        self.label_2_id_dict = self.label_2_id()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        audio, sample_rate = self.resample_audio_torchaudio(file_path)
        label = self.extract_label(file_path)
        audio = self.pad_audio(audio)
        
        return {'input_values':[audio],
                #'file':file_path,
                'sample_rate':sample_rate,
                'label':self.label_2_id_dict[label]}

    def resample_audio_torchaudio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform / waveform.abs().max()
        
        return waveform.squeeze(), self.target_sample_rate
    

        
    def extract_label(self, file_path):
        file_name = os.path.basename(file_path)
        label = file_name.split(' ')[0]
        return label
    
    def pad_audio(self, audio):
        if audio.size(0) > self.max_length:
            audio = audio[:self.max_length]
        else:
            pad_size = self.max_length - audio.size(0)
            audio = torch.nn.functional.pad(audio, (0, pad_size))
        return audio
    
    def labels(self):
        return list(set([self.extract_label(file_path) for file_path in self.file_list]))
    
    def all_labels(self):
        return [self.extract_label(file_path) for file_path in self.file_list]
    
    def id_2_label(self):
        return {i: label for i, label in enumerate(self.label_list)}
    
    def label_2_id(self):
        return {label: i for i, label in enumerate(self.label_list)}
    

    def __repr__(self) -> str:
        audio_length = len(self[0]['input_values'][0])
        duration = audio_length / self.target_sample_rate


        hours = round(len(self) * duration / (60**2),3)


        return f"AudioDataset with {len(self)} samples of {duration:.2f} seconds each\nTotal audio: {hours} hours"
    
    



parent_dir = 'data/mp3_train_files'
file_list = [os.path.join(root, file) 
             for root, _, files in os.walk(parent_dir) 
             for file in files]

random.seed(42)
random.shuffle(file_list)

train_size = int(0.8 * len(file_list))
val_size = int(0.1 * len(file_list))
test_size = len(file_list) - train_size - val_size

train_files = file_list[:train_size]
val_files = file_list[train_size:train_size + val_size]
test_files = file_list[train_size + val_size:]

train_dataset = AudioDataset(train_files)
val_dataset = AudioDataset(val_files)
test_dataset = AudioDataset(test_files)


@dataclass
class AudioDatasetSplits():
    train: AudioDataset = train_dataset
    val: AudioDataset = val_dataset
    test: AudioDataset = test_dataset

    num_classes:int = len(train.labels())
