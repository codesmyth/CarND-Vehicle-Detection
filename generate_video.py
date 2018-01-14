from moviepy.editor import VideoFileClip
from image_generation import process_image

output = './output_images/challenge.mp4'

clip1 = VideoFileClip("./challenge_video.mp4")
alf_clip = clip1.fl_image(process_image)
alf_clip.write_videofile(output, audio=False)