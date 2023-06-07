from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import sys
sys.path.append('/path/to/ffmpeg')

import numpy as np
import cv2
import librosa


class speech_dataset(Dataset):
    def __init__(self, video, sub_timestamps):
        self.mfcc = []
        self._get_segmented_audio(video, sub_timestamps)
        self._load_and_process_audio(sub_timestamps)

    def _get_segmented_audio(self, video, sub_timestamps):
        raw_audio = AudioSegment.from_mp3(video.audio_path)
        self.cache_path = video.audio_path.rsplit('\\', 1)[0]
        for i, start_ms, end_ms in sub_timestamps:
            numbering = str(i+1)
            raw_audio[start_ms:end_ms].export('./videos/segment_'+numbering+'.mp3',
                                              format="mp3")

    def _load_and_process_audio(self, sub_timestamps):
        scaler = StandardScaler()
        for i in range(1, 1 + len(sub_timestamps)):
            wav, sr = librosa.load('./videos/segment_'+str(i)+'.mp3', sr=48000)
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mfcc=64)
            mfcc = cv2.resize(mfcc, dsize=(512, 64), interpolation=cv2.INTER_AREA)
            mfcc = np.expand_dims(scaler.fit_transform(mfcc), axis=0)
            self.mfcc.append(mfcc)
        self.mfcc = np.array(self.mfcc)

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, idx):
        return self.mfcc[idx]
