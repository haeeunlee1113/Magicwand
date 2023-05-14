import moviepy.editor as mp
from model.editor.video import Video
from model.ser.ACRNN import ACRNN
from model.ser.data import speech_dataset
import torch
from torch.utils.data import TensorDataset
import stable_whisper
import numpy as np
import torch.nn as nn


class MagicWand:
    def __init__(self,):
        # model_path
        ser_path = 'C:\Users\tasty\model.pth'
        stt_path = 'C:\Users\tasty\medium.pt'

        # cuda initialization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != 'cuda':
          raise Exception("device not been connected to cuda")

        # load model
        self.ser_model = ACRNN().to(self.device)
        self.ser_model.load_state_dict(torch.load(ser_path))
        self.ser_model.eval()
        self.stt_model = stable_whisper.load_model(download_root=stt_path)
        print(f'MagicWand been initiated')
  
    def _write_srt(self, video, word_level):
        video.script.to_srt_vtt('script.srt', word_level=word_level)

    def _extract_features(self,):
        mfcc = TensorDataset(torch.from_numpy(speech_dataset()[:]))
        return mfcc

    def run_stt(self, video, word_level=True):
        transcribe_options = dict(task="transcribe", **dict(beam_size=5, best_of=5))
        video.script = self.stt_model.transcribe(video.audio_path, **transcribe_options)
        self._write_srt(video, word_level=word_level)

    def run_ser(self, video, thres=0.8):
        self.target = {}
        features = self._extract_features(video)
        softmax = nn.Softmax(dim=1)
        for i, x in enumerate(features):
            x = x.to(self.device)
            output = self.ser_model(x)
            pred_prob, pred = torch.max(softmax(output.data), 1)
            mask = pred_prob > thres
            
        self.target            

    def encode_subtitle(self, video):
        
        return 