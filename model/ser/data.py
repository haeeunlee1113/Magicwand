from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2
import librosa

class sample_dataset(Dataset):
    def __init__(self):
        self.mfcc = []
        mfcc = self.load_and_process_audio(i)
        self.mfcc.append(mfcc)
        self.mfcc = np.array(self.mfcc)
        
    def load_and_process_audio(self, num):
        wav, sr = librosa.load(f'/content/drive/MyDrive/AUDIO_DATA/attorney_sample/audio{num}.wav', sr=48000)
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mfcc=64)
        mfcc = cv2.resize(mfcc, dsize=(768, 64), interpolation=cv2.INTER_AREA)
        scaler = StandardScaler()
        mfcc = scaler.fit_transform(mfcc)
        mfcc = np.expand_dims(mfcc, axis=0)
        return mfcc

    def __len__(self):
        return len(self.mfcc)
        
    def __getitem__(self, idx):
        return self.mfcc[idx]