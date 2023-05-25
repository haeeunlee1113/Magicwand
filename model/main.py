# examples
from editor.editor import MagicWand
from editor.video import Video

# 1) video & user-defined parameters been uploaded
raw_path = "C:\\Users\\tasty\\sample.mp4"

# 2) instantiate MagicWand Model
magicwand = MagicWand()

# 3) instantiate Video class
video = Video(raw_path)

# 4) extract text from segmented speech
video = magicwand.remove_silence(video)

# 4) extract text from segmented speech
video = magicwand.run_stt(video)

# 5) recoginize emotions from segemented speech
emotion = magicwand.run_ser(video)

# 6) encode subtitles based on recognized emotions & return the result
magicwand.encode_subtitle(video, emotion)