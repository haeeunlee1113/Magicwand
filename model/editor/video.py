import moviepy.editor as mp


class Video:
    def __init__(self, video_path:str):
        # initialization
        _, ext = video_path.split('.')
        if ext not in ['ogv', 'mp4', 'mpeg', 'avi', 'mov']:
            raise Exception("Unsupported video format!")
        
        self._video_path = video_path
        self._video = mp.VideoFileClip(self._video_path)
        
    @property
    def video(self):
        return self._video
    
    @video.setter
    def video(self, vid):
        self._video = vid
    
    @property
    def video_path(self):
       return self._video_path

    @video_path.setter
    def video_path(self, path):
       self._video_path = path

    @property
    def audio_path(self):
        return self._audio_path
    
    @audio_path.setter
    def audio_path(self, path):
        self._audio_path = path

    @property
    def script(self,):
        return self._script

    @script.setter
    def script(self, file):
        self._script = file