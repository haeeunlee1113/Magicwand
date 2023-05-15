# examples
from model.editor.editor import MagicWand
from model.editor.video import Video

# 1) video & user-defined parameters been uploaded
raw_path = 'C:\Users\tasty\MagicWand\video_samples\clip.mp4'

start_time = None
end_time = None
remove_silence = True

# 2) instantiate Video class
video = Video(raw_path, start_time, end_time, remove_silence)

# 3) instantiate MagicWand Model
magicwand = MagicWand()

# 4) extract text from segmented speech
magicwand.run_stt(video)

# 5) recoginize emotions from segemented speech
magicwand.run_ser(video)

# 6) encode subtitles based on recognized emotions & return the result
result = magicwand.encode_subtitle(video)