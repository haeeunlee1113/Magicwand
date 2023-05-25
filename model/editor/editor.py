import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import moviepy.editor as mp
import re
from pydub import AudioSegment
from pycaption import SRTReader

import torch
from torch.utils.data import TensorDataset, DataLoader
import stable_whisper
from ser.data import speech_dataset
from ser.ACRNN import ACRNN

from moviepy.video.tools.subtitles import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip



class MagicWand:
    def __init__(self,):
        # model_path
        ser_path = 'C:/Users/tasty/model.pth'
        self.cache_path = 'C:\\Users\\tasty\\MagicWand\\Magicwand\\model\\cache'

        # cuda initialization
        self.device = 'cpu'

        # load model
        self.ser_model = ACRNN().to(self.device)
        self.ser_model.load_state_dict(torch.load(ser_path, map_location=self.device))
        self.ser_model.eval()
        self.stt_model = stable_whisper.load_model(name='small')
        self._empty_cache()
        print(f'MagicWand been initiated')

    def _empty_cache(self,):
        for file in os.listdir(self.cache_path):
            os.remove(f'{self.cache_path}\\{file}')
        print(f'every file in cache been removed!')
    
    def _extract_audio(self, video):
        video.video.audio.write_audiofile(f'{self.cache_path}\\raw_audio.mp3')
        audio = AudioSegment.from_mp3(f'{self.cache_path}\\raw_audio.mp3')
        return audio
    
    def _get_colors(self, max_color, min_color, pctg, thres):
        r1, g1, b1 = min_color
        r2, g2, b2 = max_color
        pctg = (pctg-thres)/(1-thres)
        r = int(r1 + (r2-r1) * pctg)
        g = int(g1 + (g2-g1) * pctg)
        b = int(b1 + (b2-b1) * pctg)
        return (r,g,b)

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
                    smin, emin = map(int, [start_time[3:].rsplit(':',1)[0], end_time[3:].rsplit(':',1)[0]])
                    sub_timestamps.append((index, 60000*smin+1000*ssec+sms, 60000*emin+1000*esec+ems))
        return sub_timestamps

    def remove_silence(self, video, thres=-35, min_silence_len=250):
        # extract the audio_file from raw audio
        raw_audio = self._extract_audio(video)
                
        # remove silence from audio
        audio = AudioSegment.empty()
        non_silence = detect_nonsilent(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len)
        segments = split_on_silence(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len, keep_silence=40)
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

    def run_ser(self, video, thres=0.75):
        sub_timestamps = self._extract_timestamps()
        features = self._extract_mfcc(video, sub_timestamps)
        
        for x in DataLoader(features, shuffle=False, batch_size=len(features)):
            output = self.ser_model(x[0].to(self.device))
            pred_prob, pred = torch.max(output.data, 1)
            emo_prob = {index: [pred[index].item(), pred_prob[index].item()]
                        if pred_prob[index].item() >= thres else None for index in range(len(features))}
        return emo_prob

    def run_stt(self, video):
        video.script = self.stt_model.transcribe(video.audio_path, task='transcribe')
        self._write_srt(video)
        return video

    def encode_subtitle(self, video, emo_prob):
        # 0:ANGRY 1:HAPPY 2:NEUTRAL 3:SAD
        subtitles = []
        emo_color = {0:[(255,0,0),(255,90,90)], 1:[(255,255,0), (255,220,30)], 2:(255,255,255), 3:[(0,84,255), (72,156,255)]}

        with open(f"{self.cache_path}\\subscript.srt", 'r', encoding='utf-8') as f:
            captions = SRTReader().read(f.read(), lang='ko')
                
        for idx, line in enumerate(captions.get_captions('ko')):
            if emo_prob[idx]:
                if emo_prob[idx][0] == 2:
                    txt = TextClip(line.get_text(), fontsize=16, font='Malgun-Gothic', color=f'rgba{emo_color[2]}')
                else:
                    txt = TextClip(line.get_text(), fontsize=16, font='Malgun-Gothic', 
                                   color=f'rgba{self._get_colors(emo_color[emo_prob[idx][0]][0], emo_color[emo_prob[idx][0]][1], emo_prob[idx][1], thres=0.75)}')
            else:
                txt = TextClip(line.get_text(), fontsize=16, font='Malgun-Gothic', color=f'rgba{(255, 255, 255)}')
            txt = txt.set_position(('center', 'bottom'))
            sub = txt.set_start(line.start/1000.).set_duration((line.end - line.start)/1000.)
            subtitles.append(sub)

        final_video = CompositeVideoClip([video.video] + subtitles)
        final_video.write_videofile(f'{self.cache_path}\\encoded_video.mp4', fps=24, threads=16, logger=None, codec="mpeg4", preset="slow", ffmpeg_params=['-b:v','10000k'])