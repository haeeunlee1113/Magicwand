import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
import moviepy.editor as mp


class Video:
    def __init__(self, video_path:str, start_time=None, end_time=None, remove_silence=True):
        # initialization
        _, ext = video_path.split('.')
        if ext not in ['ogv', 'mp4', 'mpeg', 'avi', 'mov']:
          raise Exception("Unsupported video format!")
        
        self._video_path = video_path
        self._video = mp.VideoFileClip(self._video_path, )

        # 1) get clipped video if start_time is given
        if start_time:
          self._get_clip(start_time, end_time)

        # 2) extract the audio & remove silence if remove_silence is True
        if remove_silence:
          self._remove_silence()

    @property
    def video(self):
        return self._video
    
    @property
    def video_path(self):
       return self._video_path

    @property
    def script(self,):
        return self._script

    @script.setter
    def script(self, txt):
        self._script = txt

    def _get_clip(self, start_time, end_time):
        self._video = self._video.subclip(start_time, end_time)
        self._video.write_videofile(self._video_path)
      
    def _extract_audio(self):
        self._video.audio.write_audiofile('raw_audio.mp3')
        return AudioSegment.from_file('raw_audio.mp3')
    
    def _remove_silence(self, thres=-40, min_silence_len=200):
        # extract the audio_file from raw audio
        raw_audio = self._extract_audio()
        
        # remove silence from audio
        audio = AudioSegment.empty()
        non_silence = detect_nonsilent(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len)
        segments = split_on_silence(raw_audio, silence_thresh=thres, min_silence_len=min_silence_len, keep_silence=40)
        for i, segment in enumerate(segments):
            segment.write_audiofile(f'C:\Users\tasty\MagicWand\Magicwand\model\cache\audio_segment{i}')
            audio += segment            

        # remove silence from video
        clips = []
        for (start, end) in non_silence:
          clips.append(self._video.subclip(start/1000, end/1000))
        clip = mp.concatenate_videoclips(clips, method='compose')

        # Export the final clip as a new MP4 file
        clip.write_videofile("sample.mp4")
        audio.export(self._audio_path, format='mp3')

        # remove the original file
        os.remove(self._video_path)
        self._video_path = "edited.mp4"