import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

#thx gpt4

input_folder = 'songs'
output_folder = 'converted'

os.makedirs(output_folder, exist_ok=True)

def convert_to_mp3(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format='mp3')
        print(f'Converted: {input_path} -> {output_path}')
    except Exception as e:
        print(f'Error converting {input_path}: {e}')


def extract_audio_from_mp4(input_path, output_path):
    try:
        video = VideoFileClip(input_path)
        audio_path = input_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path)
        convert_to_mp3(audio_path, output_path)
        os.remove(audio_path)
    except Exception as e:
        print(f'Error extracting audio from {input_path}: {e}')


for root, dirs, files in os.walk(input_folder):
    for filename in files:
        input_path = os.path.join(root, filename)
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.mp3')

        if filename.lower().endswith('.mp4'):
            extract_audio_from_mp4(input_path, output_path)
        elif filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac')):
            convert_to_mp3(input_path, output_path)

print('Conversion process completed.')
