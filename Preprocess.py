import subprocess
import os

input_dir = 'Features_Extractor_Videos'  # Replace with your input directory path
output_dir = 'PreprocessedFrames'  # Replace with your output directory path

def shortenVideos():
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all mp4 files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    # Loop through all files
    for video_file in video_files:
        input_file = os.path.join(input_dir, video_file)
        output_file = os.path.join(output_dir, 'preprocessed' + video_file)

        # Construct the ffmpeg command
        cmd = ['ffmpeg', '-i', input_file, '-ss', '00:00:03', '-t', '00:00:06', output_file]

        # Run the command
        subprocess.run(cmd, check=True)


def shortenVideosAndFixFrames():
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all mp4 files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    # Loop through all files
    for video_file in video_files:
        input_file = os.path.join(input_dir, video_file)
        output_file = os.path.join(output_dir, 'preprocessed' + video_file)

        # Construct the ffmpeg command
        cmd = ['ffmpeg', '-i', input_file, '-t', '1', '-r', '24', output_file]

        # Run the command
        subprocess.run(cmd, check=True)


shortenVideosAndFixFrames()