from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

import numpy as np
import cv2
import librosa
import re
import os

class speech_dataset(Dataset):
    def __init__(self, sub_timestamps):
        self.mfcc = []
        self._get_segmented_audio(sub_timestamps)
        self.load_and_process_audio(sub_timestamps)

    def _get_segmented_audio(self, sub_timestamps):
        raw_audio = AudioSegment.from_file('model/cache/audio.mp3')
        for i, timestamps in sub_timestamps:
            start_time, end_time = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', timestamps)
            raw_audio[start_time:end_time].export(f'model/cache/segments/segment_{i+1}.mp3', format="mp3")

    def _load_and_process_audio(self, sub_timestamps):
        scaler = StandardScaler()
        for i in range(1, 1+len(sub_timestamps)):
            wav, sr = librosa.load(f'/model/cache/segments/segment_{i}.mp3', sr=48000)
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mfcc=64)
            mfcc = cv2.resize(mfcc, dsize=(768, 64), interpolation=cv2.INTER_AREA)
            mfcc = scaler.fit_transform(mfcc)
            mfcc = np.expand_dims(mfcc, axis=0)
            self.mfcc.append(mfcc)
        self.mfcc = np.array(self.mfcc)

    def __len__(self):
        return len(self.mfcc)
        
    def __getitem__(self, idx):
        return self.mfcc[idx]