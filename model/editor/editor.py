import torch
from torch.utils.data import TensorDataset
import stable_whisper
import numpy as np
import moviepy.editor as mp

from model.ser.data import speech_dataset
from model.editor.video import Video
from model.ser.ACRNN import ACRNN


class MagicWand:
    def __init__(self,):
        # model_path
        ser_path = 'C:/Users/tasty/model.pth'
        stt_path = 'C:/Users/tasty/medium.pt'
        self.cache_path = 'C:/Users/tasty/MagicWand/Magicwand/model/cache'

        # cuda initialization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load model
        self.ser_model = ACRNN().to(self.device)
        self.ser_model.load_state_dict(torch.load(ser_path))
        self.ser_model.eval()
        self.stt_model = stable_whisper.load_model(name='medium', download_root=stt_path)
        print(f'MagicWand been initiated')
  
    def _write_srt(self, video):
        video.script.to_srt_vtt(f'{self.cache_path}/subscript.srt', segment_level = True, word_level = False) 

    def _extract_features(self, script):
        features = speech_dataset(script)
        mfcc = TensorDataset(torch.from_numpy(features[:]))
        return mfcc
    
    def _extract_timestamps(self, script):
        with open(f'{self.cache_path}/subscript.srt', 'r') as f:
            lines f.read().split('\n\n')
            subtitles = []
            for line in lines:
                parts = line.split('\n')
                if len(parts) >= 3:
                    index = parts[0]
                    timestamps = parts[1]
                    text = ' '.join(parts[2:])
                    subtitles.append((index, timestamps, text))
        return subtitles

    def run_ser(self, video, thres=0.8):
        features = self._extract_features(video.script)
        
        for i, x in enumerate(features):
            x = x.to(self.device)
            output = self.ser_model(x)
            pred_prob, pred = torch.max(softmax(output.data), 1)
            mask = pred_prob > thres    

    def run_stt(self, video):
        transcribe_options = dict(task="transcribe", **dict(beam_size=5, best_of=5))
        video.script = self.stt_model.transcribe(video.audio_path, **transcribe_options)
        self._write_srt(video)

    def encode_subtitle(self, video):
        
        return 