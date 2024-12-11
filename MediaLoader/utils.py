import cv2
import sys
import os
from moviepy.editor import VideoFileClip

def get_num_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return num_frames

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_codec(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()
    return codec

def get_width(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return width

def get_height(video_path):
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height

def get_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return duration

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return num_frames, fps, codec, width, height, duration

def get_video_info_dict(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {'num_frames': num_frames, 'fps': fps, 'codec': codec, 'width': width, 'height': height, 'duration': duration}

def encode_video_H264(video_path, remove_original=False):
    """
    Encode video to H264 codec
    
    Args:
        video_path (str): path to video to be encoded
        remove_original (bool): whether to remove original video after encoding
    Returns:
        output_path (str): path to encoded video
    """

    output_path = video_path.split('.')[0] + '_H264.mp4'
    clip = VideoFileClip(video_path)
    clip.write_videofile(output_path, codec='libx264')
    if remove_original:
        os.remove(video_path)
    clip.close()

    return output_path

def cut_video(video_path, start_frame, end_frame, output_path):
    """
    Cut video from start_frame to end_frame

    Args:
        video_path (str): path to video to be cut
        start_frame (int): start frame
        end_frame (int): end frame
        output_path (str): path to output video
        
    Returns:
        output_path (str): path to output video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == end_frame:
                break
        else:
            break
    
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    
    return output_path

def cut_video_moviepy(video_path, start_frame, end_frame, output_path):
    """
    Cut video from start_frame to end_frame using MoviePy

    Args:
        video_path (str): path to video to be cut
        start_frame (int): start frame
        end_frame (int): end frame
        output_path (str): path to output video
        
    Returns:
        output_path (str): path to output video
    """
    video = VideoFileClip(video_path)
    fps = video.fps
    duration = video.duration

    # Ensure start and end frames are within valid range
    start_frame = max(0, min(start_frame, duration * fps))
    end_frame = max(start_frame, min(end_frame, duration * fps))

    cut_video = video.subclip(start_frame / fps, end_frame / fps)
    cut_video.write_videofile(output_path, codec='libx264')

    return output_path


if __name__ == "__main__":
    video_path = os.path.join("outputs", "visualizations", "fail_H264.mp4")

    cut_video(video_path, 0, 50, video_path.split('.')[0] + '_fail_1.mp4')
    
