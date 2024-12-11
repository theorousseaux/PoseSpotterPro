import cv2
import time
import torch
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nms
from mmengine.registry import init_default_scope
from mmpose.registry import VISUALIZERS
import mmcv
from mmpose.utils import adapt_mmdet_pipeline
import os
import sys
sys.path.append(os.getcwd())
from MediaLoader.utils import get_video_info_dict

DATASET = ['CAUCAU', 'cropped']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device : ", device)

# model to use
pose_config = 'mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'

# build pose estimator
pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device)

# build visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.alpha = 0.8
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(
    pose_estimator.dataset_meta, skeleton_style='openpose')

def process_one_image_with_bbox(img,
                      pose_estimator, bboxes,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style='openpose',
            show=False,
            wait_time=show_interval,
            kpt_thr=0.3)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def instances_list_from_preds(preds):
    """
    Convert the predictions to a list of instances

    Args:
        preds (dict): the predictions

    Returns:
        instances_list (list): the list of instances
    """
    instances_list = []
    pred_dict = preds.to_dict()
    for i in range(len(pred_dict['bboxes'])):
        instance = {}
        instance['keypoints'] = pred_dict['keypoints'][i].astype('float64').tolist()
        instance['keypoint_scores'] = pred_dict['keypoint_scores'][i].astype('float64').tolist()
        instance['bbox'] = [pred_dict['bboxes'][i].astype('float64').tolist()]
        instance['bbox_score'] = pred_dict['bbox_scores'][i].astype('float64')
        instances_list.append(instance)
    
    return instances_list


def get_video_path(path_imgs):
    path_split = path_imgs.split("/")
    subject_number = path_split[-3].split(".")[-1]
    action_split = path_split[-2].split(" ")
    if len(action_split) == 2:
        action_split[1] = action_split[1].capitalize()
    path_video = os.sep.join(path_split[:-3]) + '/video/' + ''.join(action_split) + 'S' + subject_number + '.avi'
    return path_video

def process_CAUCAU_video_with_bbox(path_imgs, output_directory, video_name):

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory + "visualizations", exist_ok=True) 
    os.makedirs(output_directory + "predictions", exist_ok=True) 

    # to save the prediction
    pose_sequence = []

    # Get the video info
    path_video = path_imgs
    path_split = path_imgs.split("/")
    subject_number = path_split[-3].split(".")[-1]
    action_split = path_split[-2].split(" ")
    if len(action_split) == 2:
        action_split[1] = action_split[1].capitalize()

    path_video = os.sep.join(path_split[:-3]) + '/video/' + ''.join(action_split) + 'S' + subject_number + '.avi'
    info_dict = get_video_info_dict(path_video)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writter = cv2.VideoWriter(output_directory + "visualizations/" + video_name, fourcc, 30, (info_dict['width'], info_dict['height']))

    # Get the list of images to process
    imgs_list = os.listdir(path_imgs)
    imgs_list.sort()

    num_frames = 0

    imgs_list = os.listdir(path_imgs)
    imgs_list.sort()

    start_time = time.time()
    for frame_number, input in enumerate(imgs_list):
        frame_dict = {}
        frame_dict['frame_id'] = frame_number

        if input.endswith(".txt"):
            continue

        try:
            img_path = path_imgs + input
            img_pil = Image.open(img_path)
            img = np.array(img_pil)
            # imposed bbox
            with open(img_path.replace("png", "txt"), 'rb') as f:
                bbox_yolo = np.loadtxt(f, delimiter=' ', dtype=np.float32)[1:] # YOLO format : [x-center, y-center, width, height]
                # We convert it to the COCO format : [xmin, ymin, xmax, ymax]
                bbox = np.zeros(4)
                bbox[0] = (bbox_yolo[0] - bbox_yolo[2] / 2) * img_pil.width
                bbox[1] = (bbox_yolo[1] - bbox_yolo[3] / 2) * img_pil.height
                bbox[2] = (bbox_yolo[0] + bbox_yolo[2] / 2) * img_pil.width
                bbox[3] = (bbox_yolo[1] + bbox_yolo[3] / 2) * img_pil.height
            bboxes = np.expand_dims(bbox, axis=0)
            # topdown pose estimation
            preds = process_one_image_with_bbox(img,
                                                pose_estimator, bboxes, visualizer,
                                                0.000)
            frame_dict['instances'] = instances_list_from_preds(preds)
            img_output = visualizer.get_image()
        except:
            print("Fail")

        num_frames += 1
        video_writter.write(img_output)
        pose_sequence.append(frame_dict)
    
    with open(output_directory + "predictions/" + video_name.replace(".avi", ".json"), 'w') as f:
        json.dump(pose_sequence, f)

    fps = num_frames / (time.time() - start_time)
    print("FPS : ", fps)


if __name__ == '__main__':

    dataset = "CAUCAFall"

    if dataset == "CAUCAFall":
        # to save the prediction
        dict_list = []

        subject_list = os.listdir("Data/Fall/Dataset_CAUCAFall/CAUCAFall")
        subject_list.sort()
        subject_list.remove("video")
        for subject_number in subject_list:
            action_list = os.listdir("Data/Fall/Dataset_CAUCAFall/CAUCAFall/" + subject_number)
            action_list.sort()
            for action in action_list:
                path_imgs = "Data/Fall/Dataset_CAUCAFall/CAUCAFall/" + subject_number + "/" + action + "/"

                action_split = action.split(" ")
                if len(action_split) == 2:
                    action_split[1] = action_split[1].capitalize()
                action_rename = ''.join(action_split)
                process_CAUCAU_video_with_bbox(path_imgs, output_directory='outputs/fall/bbox_imposed/', video_name=action_rename + "S" + subject_number.split(".")[-1] + '.avi')

                print("Done with " + subject_number + " " + action)

