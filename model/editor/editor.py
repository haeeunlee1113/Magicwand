import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import moviepy.editor as mp
import re
from pydub import AudioSegment
import pysubs2

import torch
from torch.utils.data import TensorDataset, DataLoader
import stable_whisper 
from ser.data import speech_dataset
from ser.ACRNN import ACRNN
import subprocess


class MagicWand:
    def __init__(self):
        # model_path
        ser_path = 'C:/Users/tasty/model.pth'
        self.cache_path = 'C:\\Users\\tasty\\MagicWand\\Magicwand\\model\\cache'

        # cuda initialization
        self.device = 'cpu'

        # load model
        self.ser_model = ACRNN().to(self.device)
        self.ser_model.load_state_dict(torch.load(ser_path, map_location=self.device))
        self.ser_model.eval()
        self.stt_model = stable_whisper.load_model('medium')
        self._empty_cache()
        print(f'MagicWand been initiated')

    def _empty_cache(self,):
        for file in os.listdir(self.cache_path):
            os.remove(f'{self.cache_path}\\{file}')
        print(f'every file in cache been removed!')
    
    def _extract_audio(self, video, flag=False):
        video.video.audio.write_audiofile(f'{self.cache_path}\\audio.mp3')
        if flag:
            video.audio_path = f'{self.cache_path}\\audio.mp3'
            return video
        return AudioSegment.from_mp3(f'{self.cache_path}\\audio.mp3')
    
    def _get_colors(self, max_color, min_color, pctg, thres):
        r1, g1, b1 = min_color
        r2, g2, b2 = max_color
        pctg = (pctg-thres)/(1-thres)
        r = int(r1 + (r2-r1) * pctg)
        g = int(g1 + (g2-g1) * pctg)
        b = int(b1 + (b2-b1) * pctg)
        return pysubs2.Color(r,g,b)

    def _write_srt(self, video):
        video.script.to_srt_vtt(f'{self.cache_path}\\subscript.srt', segment_level = True, word_level = False) 
        print(f'script been successfully converted to srt!')
        return video
    
    def _extract_mfcc(self, video, timestamps):
        features = speech_dataset(video, timestamps)
        mfcc = TensorDataset(torch.from_numpy(features[:]))
        return mfcc
    
    def _extract_timestamps(self):
        with open(f'{self.cache_path}\\subscript.srt', 'r', encoding='utf-8') as f:
            lines = f.read().split('\n\n')
            sub_timestamps = []
            for line in lines:
                parts = line.split('\n')
                if len(parts) >= 3:
                    index = int(parts[0])
                    start_time, end_time = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', parts[1])
                    ssec, sms = map(int, start_time[3:].rsplit(':',1)[1].split(','))
                    esec, ems = map(int, end_time[3:].rsplit(':',1)[1].split(','))
                    smin, emin = map(int, [start_time[3:].split(':',1)[0], end_time[3:].split(':',1)[0]])
                    sub_timestamps.append((index, 60000*smin+1000*ssec+sms, 60000*emin+1000*esec+ems))
        return sub_timestamps

    def remove_silence(self, video, thres=-40, min_silence_len=500):
        # extract the audio_file from raw audio
        raw_audio = self._extract_audio(video)
                
        # remove silence from audio
        audio = AudioSegment.empty()
        non_silence = detect_nonsilent(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len)
        segments = split_on_silence(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len, keep_silence=0)
        for segment in segments:
            audio += segment

        # remove silence from video
        clips = []
        for (start, end) in non_silence:
          clips.append(video.video.subclip(start/1000, end/1000))
        clip = mp.concatenate_videoclips(clips, method='compose')

        # Export the final clip as a new MP4 file
        clip.write_videofile(f"{self.cache_path}\\clip.mp4")
        audio.export(f"{self.cache_path}\\audio.mp3", format='mp3')        
        video.video = clip
        video.video_path = f"{self.cache_path}\\clip.mp4"
        video.audio_path = f'{self.cache_path}\\audio.mp3'
        return video

    def run_ser(self, video, thres=0.7):
        sub_timestamps = self._extract_timestamps()
        features = self._extract_mfcc(video, sub_timestamps)
        
        for x in DataLoader(features, shuffle=False, batch_size=len(features)):
            output = self.ser_model(x[0].to(self.device))
            pred_prob, pred = torch.max(output.data, 1)
            emo_prob = {index: [pred[index].item(), pred_prob[index].item()]
                        if pred_prob[index].item() >= thres else None for index in range(len(features))}
        return emo_prob

    def run_stt(self, video):
        if video.audio_path == None:
            video = self._extract_audio(video, flag=True)

        video.script = self.stt_model.transcribe(video.audio_path, word_timestamps=True)
        self._write_srt(video)
        return video


    def encode_subtitle(self, video, emo_prob):
        # 0:ANGRY 1:HAPPY 2:NEUTRAL 3:SAD
        emo_color = {0:[(255,0,0),(255,90,90)], 1:[(255,255,0), (255,220,30)], 2:(255,255,255), 3:[(0,84,255), (72,156,255)]}

        subs = pysubs2.load(f"{self.cache_path}\\subscript.srt")
        
        captions = pysubs2.SSAFile()
        captions.info['ScaledBorderAndShadow'] = 'no'
        captions.info['YCbCr Matrix'] = 'TV.601'
        captions.styles = dict()

        def _get_style(emo):
            if emo[0] == 0:
                return pysubs2.SSAStyle(fontname='Mapo홍대프리덤', fontsize=40, shadow=0, outline=0, 
                                        alignment=pysubs2.Alignment.MIDDLE_CENTER, primarycolor=self._get_colors(emo_color[0][0], emo_color[0][1], emo[1], 0.7))
            elif emo[0] == 1:
                return pysubs2.SSAStyle(fontname='스스로넷 칠백삼', fontsize=40, shadow=0, outline=0, 
                                        alignment=pysubs2.Alignment.MIDDLE_CENTER, primarycolor=self._get_colors(emo_color[1][0], emo_color[1][1], emo[1], 0.7))
            elif emo[0] == 2:
                return pysubs2.SSAStyle(fontname='Pretendard', fontsize=40, shadow=0, outline=0, 
                                        alignment=pysubs2.Alignment.MIDDLE_CENTER, primarycolor=pysubs2.Color(255,255,255))
            else:
                return pysubs2.SSAStyle(fontname='완도희망체', fontsize=40, shadow=0, outline=0, 
                                        alignment=pysubs2.Alignment.MIDDLE_CENTER, primarycolor=self._get_colors(emo_color[3][0], emo_color[3][1], emo[1], 0.7))

        for idx, line in enumerate(subs):
            if emo_prob[idx]:
                captions.styles[f'style_{idx}'] = _get_style(emo_prob[idx])
                line.style = f'style_{idx}'
                captions.append(line)
            else:
                captions.styles[f'style_{idx}'] = pysubs2.SSAStyle(fontname='Pretendard', fontsize=80, shadow=0, outline=0, 
                                                                   alignment=pysubs2.Alignment.MIDDLE_CENTER, primarycolor=pysubs2.Color(255,255,255))
                line.style = f'style_{idx}'
                captions.append(line)

        captions.save(f'{self.cache_path}\\subscript.ass', format='ass')
        
        ffmpeg_cmd = f'ffmpeg -i {video.video_path} -vf "subtitles=model/cache/subscript.ass" -c:a copy {self.cache_path}\\encoded_video.mp4'
        subprocess.call(ffmpeg_cmd, shell=True)