import subprocess
import os
import json

def add_text_to_center(input_file, output_file, text):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'drawtext=text={text}:x=(w-text_w)/2:y=40:fontsize=128:fontcolor=black:box=1:boxcolor=white@0.75:boxborderw=5',
        '-c:a', 'copy',
        output_file
    ]

    subprocess.run(command)

def resize_video(input_file, output_file):
    """
    Reduce the height and width of a video by half.
    
    Parameters:
    - input_file: Path to the source .mp4 file.
    - output_file: Path to the output .mp4 file.
    """
    
    if not os.path.exists(input_file):
        raise ValueError(f"The file {input_file} does not exist.")
    
    # Command to reduce the video dimensions by half
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", "scale=iw/2:ih/2",  # iw refers to input width, and ih refers to input height
        "-c:v", "libx264",
        "-c:a", "copy",  # copying the audio stream without re-encoding
        output_file
    ]
    
    subprocess.run(cmd, check=True)

def print_video_info(input_file):
    """
    Print information about a video file.
    
    Parameters:
    - input_file: Path to the video file.
    """
    
    # Use ffprobe to gather video information in JSON format
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_file
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_info = json.loads(result.stdout)

    # Extracting the required details
    file_size = video_info['format']['size']
    file_format = video_info['format']['format_name']
    duration = float(video_info['format']['duration'])
    video_stream = [s for s in video_info['streams'] if s['codec_type'] == 'video'][0]
    height = video_stream['height']
    width = video_stream['width']  
    fps = eval(video_stream['r_frame_rate'])
    scan_type = video_stream.get('field_order', 'Progressive')

    # Printing the details
    print(f"File size: {file_size} bytes")
    print(f"Format: {file_format}")
    print(f"Duration: {duration} seconds")
    print(f"Height: {height} pixels")
    print(f"Width: {width} pixels")
    print(f"FPS: {fps}")
    print(f"Scan type: {scan_type}")

def increase_video_speed(input_file, output_file, speed_factor=2.0):
    """
    Increase the playback speed of a video.
    
    Parameters:
    - input_file: Path to the source video file.
    - output_file: Path to the output video file.
    - speed_factor: Factor by which to increase the speed (default is 2.0).
    """
    
    if not os.path.exists(input_file):
        raise ValueError(f"The file {input_file} does not exist.")
    
    # Calculate the setpts value for ffmpeg
    setpts_value = 1.0 / speed_factor

    # Command to increase video speed
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", f"setpts={setpts_value}*PTS",
        "-af", f"atempo={speed_factor}",
        output_file
    ]
    
    subprocess.run(cmd, check=True)

def reduce_mp4_size(input_file, output_file, video_bitrate="500k", audio_bitrate="128k"):
    """
    Reduce the size of an MP4 file by adjusting its bitrate.
    
    Parameters:
    - input_file: Path to the source .mp4 file.
    - output_file: Path to the output .mp4 file.
    - video_bitrate: Desired video bitrate (default is "500k").
    - audio_bitrate: Desired audio bitrate (default is "128k").
    """
    
    if not os.path.exists(input_file):
        raise ValueError(f"The file {input_file} does not exist.")
    
    # Command to reduce the MP4 file size
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264",
        "-b:v", video_bitrate,
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        output_file
    ]
    
    subprocess.run(cmd, check=True)

def merge_videos_2x2(video_paths, output_file):
    """
    Merge four videos into a 2x2 matrix shape.
    
    Parameters:
    - video_paths: List of paths to the 4 mp4 files.
    - output_file: Path to the output .mp4 file.
    """
    
    for video in video_paths:
        if not os.path.exists(video):
            raise ValueError(f"The file {video} does not exist.")

    # Command to combine 4 videos in a 2x2 grid
    cmd = [
        "ffmpeg",
        "-i", video_paths[0],  # top-left
        "-i", video_paths[1],  # top-right
        "-i", video_paths[2],  # bottom-left
        "-i", video_paths[3],  # bottom-right
        "-filter_complex",
        "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_file
    ]

    subprocess.run(cmd, check=True)

def convert_dav_to_mp4(input_file, output_file, compression_rate=28):
    """
    Convert .dav file to .mp4, reduce its size, and double its speed.
    
    Parameters:
    - input_file: Path to the .dav file.
    - output_file: Path to the output .mp4 file.
    - compression_rate: CRF value for the x264 codec (default is 23, the lower the value the better the quality).
    """
    
    if not os.path.exists(input_file):
        raise ValueError(f"The file {input_file} does not exist.")
    
    # Command for converting .dav to .mp4, compressing it, and doubling the speed
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", "setpts=0.5*PTS",  # This doubles the video speed
        "-c:v", "libx264",
        "-crf", str(compression_rate),
        "-c:a", "aac",
        "-strict", "experimental",
        output_file
    ]
    
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    source_file = "/Users/baha2r/PhD/Pybullet/TestingVideos/TOP/2023-09-14-11-36-50_CAM 4_1001$1$0$3.dav"
    destination_file = "TOP.mp4"
    convert_dav_to_mp4(source_file, destination_file)

    # video_files = [
    #     "SIDE.mp4",
    #     "SIDE2.mp4",
    #     "TOP.mp4",
    #     "BEHIND.mp4"
    # ]
    # destination_file = "merged_output.mp4"
    # merge_videos_2x2(video_files, destination_file)

    # source_file = "merged_output.mp4"
    # destination_file = "reduced_size_output.mp4"
    # reduce_mp4_size(source_file, destination_file)

    # source_file = "merged_output.mp4"
    # destination_file = "faster_output.mp4"
    # increase_video_speed(source_file, destination_file)

    # source_file = "faster_output.mp4"
    # destination_file = "resized_output.mp4"
    # resize_video(source_file, destination_file)

    # print_video_info(source_file)
    # print_video_info(destination_file)

    input_video = "resized_output.mp4"
    output_video = "path_to_output_video.mp4"
    text_to_add = "4x"
    add_text_to_center(input_video, output_video, text_to_add)
    print_video_info(output_video)