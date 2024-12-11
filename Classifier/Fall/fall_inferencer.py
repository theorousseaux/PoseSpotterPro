import json
import pickle
import os
import sys
import cv2
import time
import argparse
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from PoseEstimation.pose_inferencer import PoseInferencer
from Classifier.Fall.data_preparation import WindowFeatureExtractor
from MediaLoader.utils import get_video_info_dict

class FallInferencer:

    def __init__(self, pose_inferencer, window_size, step_size):
        self.pose_inferencer = pose_inferencer
        self.window_size = window_size
        self.step_size = step_size

        self.feature_extractor = WindowFeatureExtractor(window_size=self.window_size, step_size=self.step_size)
        model = pickle.load(open('Classifier/Fall/models/randomForest_window_{}_step_{}.sav'.format(self.window_size, self.step_size), 'rb'))
        scaler = pickle.load(open('Classifier/Fall/models/standardScaler_window_{}_step_{}.sav'.format(self.window_size, self.step_size), 'rb'))

        self.pipeline = Pipeline([('scaler', scaler), ('model', model)])


    def get_color_from_status(self, status):
        """Return the color corresponding to the status."""
        if status == "Fall":
            return (0, 165, 255)  # Orange
        elif status == "normal":
            return (0, 255, 0)    # Vert
        elif status == "Lying down":
            return (0, 0, 255)    # Rouge
        else:
            return (255, 255, 255)  # Blanc pour les autres statuts

    def process_one_video(self, video_path, output_directory, bboxes_list=None, show_interval=0, callback=lambda x, y, z: None):

        """
        Apply pose estimation on one video, save the results and detect fall.

        Args:
            video_path (str): Path of the video to be processed.
            output_directory (str): Path of the output directory.
            bboxes_list -optional (list[list[np.ndarray]]): Bounding boxes of human instances.
            show_interval -optional (int): Interval of visualization. If set to 0, the
                results will be shown all the time.
        """


        # Create the output directory if it doesn't exist
        os.makedirs(output_directory + "visualizations", exist_ok=True) 
        os.makedirs(output_directory + "predictions", exist_ok=True) 

        # to save the prediction
        pose_sequence = []
        info_dict = get_video_info_dict(video_path)
        video_name = video_path.split(os.sep)[-1]
        video_extension = '.' + video_name.split(".")[-1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writter = cv2.VideoWriter(output_directory + "visualizations" + os.sep + video_name, fourcc, info_dict['fps'], (info_dict['width'], info_dict['height']))

        num_frames = 0
        status = 'Normal'

        start_time = time.time()

        # read video
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            frame_dict = {}
            frame_dict['frame_id'] = num_frames

            try:
                if bboxes_list is not None:
                    bboxes = bboxes_list[num_frames]
                else:
                    bboxes = None
                # topdown pose estimation
                preds = self.pose_inferencer.infer_one_image(frame, bboxes=bboxes, show_interval=show_interval)
                frame_dict['instances'] = self.pose_inferencer.instances_list_from_preds(preds)
                img_output = self.pose_inferencer.visualizer.get_image()
            except KeyError as e:
                print(e)
                break

            num_frames += 1
            img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)   
            pose_sequence.append(frame_dict)

            # fall detection
            if num_frames >= self.window_size and (num_frames) % self.step_size == 0:
                df = self.feature_extractor.get_df_from_preds(pose_sequence[-self.window_size:], instance_id=0)
                features_df = self.feature_extractor.create_sample(df)
                preds_fall = self.pipeline.predict(features_df)
                print(preds_fall)
                status = preds_fall[-1]

            bbox = frame_dict['instances'][0]['bbox'][0]
            x_max, y_min = bbox[2], bbox[1]
            font_scale = 0.5
            (text_width, text_height), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_x = int(x_max - text_width - 10)  # -10 pour une petite marge
            text_y = int(y_min + text_height + 10)  # +10 pour une petite marge
            color = self.get_color_from_status(status)
            
            
            cv2.putText(img_output, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
            video_writter.write(img_output)
        
            fps = num_frames / (time.time() - start_time)
            callback(num_frames, fps, info_dict)

        cap.release()
        with open(output_directory + "predictions" + os.sep + video_name.replace(video_extension, ".json"), 'w') as f:
            json.dump(pose_sequence, f)

    def process_one_video_without_HPE(self, video_path, pred_path, output_directory, callback=lambda x, y, z: None):

        os.makedirs(output_directory, exist_ok=True)

        with open(pred_path) as f:
            pose_sequence = json.load(f)

        info_dict = get_video_info_dict(video_path)
        video_name = video_path.split(os.sep)[-1].split('.')[0]
        video_extension = '.' + video_path.split(".")[-1]

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writter = cv2.VideoWriter(output_directory + video_name + '_fall' + video_extension, fourcc, info_dict['fps'], (info_dict['width'], info_dict['height']))

        num_frames = 0
        status = 'Normal'

        start_time = time.time()
        # read video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            num_frames += 1

            # fall detection
            if num_frames >= self.window_size and (num_frames) % self.step_size == 0:
                df = self.feature_extractor.get_df_from_preds(pose_sequence[:num_frames], instance_id=0)
                features_df = self.feature_extractor.create_sample(df)
                preds = self.pipeline.predict(features_df)
                print(preds)
                status = preds[-1]

            bbox = pose_sequence[num_frames]['instances'][0]['bbox'][0]
            x_max, y_min = bbox[2], bbox[1]
            font_scale = 0.5
            (text_width, text_height), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_x = int(x_max - text_width - 10)  # -10 pour une petite marge
            text_y = int(y_min + text_height + 10)  # +10 pour une petite marge
            color = self.get_color_from_status(status)
            
            
            cv2.putText(frame, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
            video_writter.write(frame)

            fps = num_frames / (time.time() - start_time)
            callback(num_frames, fps, info_dict)

        cap.release()
    
        return output_directory + video_name + '_fall' + video_extension

def parse_args():
    parser = argparse.ArgumentParser(description='Fall Detection Inference')

    parser.add_argument('--output-directory', type=str, help='Path of the output directory.', required=True)
    parser.add_argument('--video-path', type=str, help='Path of the video to be processed.')
    parser.add_argument('--preds-path', type=str, help='Path of the predictions file.')
    parser.add_argument('--video-directory', type=str, help='Path of the video directory.')
    parser.add_argument('--preds-directory', type=str, help='Path of the predictions directory.')

    parser.add_argument('--window-size', type=int, help='Window size.', default=10)
    parser.add_argument('--step-size', type=int, help='Step size.', default=10)

    parser.add_argument('--detector', type=str, help='Detector.', default='RTMDetM')
    parser.add_argument('--pose-estimator', type=str, help='Pose estimator.', default='RTMPoseS')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_directory + "visualizations", exist_ok=True) 
    os.makedirs(args.output_directory + "predictions", exist_ok=True) 

    fall_inferencer = FallInferencer(pose_inferencer=PoseInferencer(detector=args.detector, pose_estimator=args.pose_estimator), window_size=args.window_size, step_size=args.step_size)

    if args.video_path is not None and args.preds_path is None:
        fall_inferencer.process_one_video(video_path=args.video_path, output_directory=args.output_directory, show_interval=0)
    
    if args.video_directory is not None and args.preds_directory is None:
        video_list = os.listdir(args.video_directory)
        for video in video_list:
            print('processing video : ', video)
            fall_inferencer.process_one_video(video_path=args.video_directory + video, output_directory=args.output_directory, show_interval=0)
    
    if args.preds_path is not None:
        fall_inferencer.process_one_video_without_HPE(video_path=args.video_path, pred_path=args.preds_path, output_directory=args.output_directory)

    if args.preds_directory is not None:
        video_list = os.listdir(args.video_directory)
        for video in video_list:
            print('processing video : ', video)
            video_extension = '.' + video.split(".")[-1]
            preds_path = args.preds_directory + video.replace(video_extension, ".json")
            fall_inferencer.process_one_video_without_HPE(video_path=args.video_directory + video, pred_path=preds_path, output_directory=args.output_directory)

if __name__ == '__main__':

    main()
