import cv2
import time
import torch
import json
import argparse
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser(description="MMPose video inference")

    parser.add_argument(
        "--output-directory", type=str, help="Path of the output directory."
    )

    parser.add_argument("--detector", type=str, default="RMTDetM", help="detector name")

    parser.add_argument(
        "--pose-estimator", type=str, default="RTMPoseM", help="pose estimator name"
    )

    parser.add_argument("--image-path", type=str, default=None, help="image file path")

    parser.add_argument("--video-path", type=str, default=None, help="video file path")

    parser.add_argument(
        "--video-directory", type=str, default=None, help="video directory"
    )

    args = parser.parse_args()

    return args


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device : ", device)

det_dict = {
    "RTMDetM": {
        "config": "PoseEstimation/models/det/rtmdet_m_640-8xb32_coco-person.py",
        "checkpoint": "PoseEstimation/models/det/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
    },
    "RTMDetNano": {
        "config": "PoseEstimation/models/det/rtmdet_nano_320-8xb32_coco-person.py",
        "checkpoint": "PoseEstimation/models/det/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth",
    },
}

pose_dict = {
    "RTMPoseM": {
        "config": "PoseEstimation/models/pose/rtmpose-m_8xb256-420e_coco-256x192.py",
        "checkpoint": "PoseEstimation/models/pose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
    },
    "RTMPoseS": {
        "config": "PoseEstimation/models/pose/rtmpose-s_8xb256-420e_coco-256x192.py",
        "checkpoint": "PoseEstimation/models/pose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth",
    },
    "RTMPoseT": {
        "config": "PoseEstimation/models/pose/rtmpose-t_8xb256-420e_coco-256x192.py",
        "checkpoint": "PoseEstimation/models/pose/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth",
    },
}


class PoseInferencer:

    def __init__(self, detector, pose_estimator, radius=3, alpha=0.8, line_width=1):

        self.detector_name = (
            det_dict[detector]["checkpoint"].split("/")[-1].split(".")[0]
        )
        self.pose_estimator_name = (
            pose_dict[pose_estimator]["checkpoint"].split("/")[-1].split(".")[0]
        )
        print(det_dict[detector]["config"])
        # build detector
        self.detector = init_detector(
            det_dict[detector]["config"],
            det_dict[detector]["checkpoint"],
            device=device,
        )
        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            pose_dict[pose_estimator]["config"],
            pose_dict[pose_estimator]["checkpoint"],
            device=device,
        )

        # build visualizer
        self.pose_estimator.cfg.visualizer.radius = radius
        self.pose_estimator.cfg.visualizer.alpha = alpha
        self.pose_estimator.cfg.visualizer.line_width = line_width
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta, skeleton_style="openpose"
        )

    def infer_one_image(self, img, bboxes=None, show_interval=0):
        """
        Apply pose estimation on one image. If bboxes is not provided, use the detector to get bboxes.

        Args:
            img (np.ndarray): Image to be processed.
            bboxes (list[np.ndarray]): Bounding boxes of human instances.
            show_interval (int): Interval of visualization. If set to 0, the
                results will be shown all the time.

        Returns:
            preds (dict): The prediction results. The keys of the dict are 'keypoints', 'bboxes', 'keypoint_scores', 'bbox_scores'.
        """

        if bboxes is None:
            # predict bbox
            scope = self.detector.cfg.get("default_scope", "mmdet")
            if scope is not None:
                init_default_scope(scope)
            detect_result = inference_detector(self.detector, img)
            pred_instance = detect_result.pred_instances.cpu().numpy()
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
            )
            bboxes = bboxes[
                np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)
            ]
            bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        # show the results
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order="rgb")
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        if self.visualizer is not None:
            self.visualizer.add_datasample(
                "result",
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap=False,
                draw_bbox=True,
                show_kpt_idx=False,
                skeleton_style="openpose",
                show=False,
                wait_time=show_interval,
                kpt_thr=0.3,
            )

        # if there is no instance detected, return None
        return data_samples.get("pred_instances", None).to_dict()

    def instances_list_from_preds(self, preds_dict):
        """
        Convert the predictions to a list of instances

        Args:
            preds (dict): the predictions

        Returns:
            instances_list (list): the list of instances
        """
        instances_list = []

        for i in range(len(preds_dict["bboxes"])):
            instance = {}
            instance["keypoints"] = (
                preds_dict["keypoints"][i].astype("float64").tolist()
            )
            instance["keypoint_scores"] = (
                preds_dict["keypoint_scores"][i].astype("float64").tolist()
            )
            instance["bbox"] = [preds_dict["bboxes"][i].astype("float64").tolist()]
            instance["bbox_score"] = preds_dict["bbox_scores"][i].astype("float64")
            instances_list.append(instance)

        return instances_list

    def process_one_image(self, img_path, output_directory, bboxes=None):
        """
        Apply pose estimation on one image and save the results.

        Args:
            img_path (str): Path of the image to be processed.
            output_directory (str): Path of the output directory.
            bboxes (list[np.ndarray]): Bounding boxes of human instances.
        """

        os.makedirs(output_directory + "visualizations", exist_ok=True)
        os.makedirs(output_directory + "predictions", exist_ok=True)

        preds = self.infer_one_image(img_path, bboxes)
        preds = self.instances_list_from_preds(preds)

        # save the prediction
        with open(
            output_directory
            + "predictions/"
            + img_path.split(os.sep)[-1].replace(img_path.split(".")[-1], "json"),
            "w",
        ) as f:
            json.dump(preds, f)

        # save the visualization
        img_output = self.visualizer.get_image()
        img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            output_directory + "visualizations/" + img_path.split(os.sep)[-1],
            img_output,
        )

    def process_one_frame(self, frame, output_directory, bboxes=None):
        """
        Apply pose estimation on one image and save the results.

        Args:
            frame (np.ndarray): Image to be processed.
            output_directory (str): Path of the output directory.
            file_name (str): Name of the output file.
            bboxes (list[np.ndarray]): Bounding boxes of human instances.

        Returns:
            preds (list): The prediction results. The keys of the dict are 'keypoints', 'bboxes', 'keypoint_scores', 'bbox_scores'.
        """

        os.makedirs(output_directory + "visualizations", exist_ok=True)
        os.makedirs(output_directory + "predictions", exist_ok=True)

        preds = self.infer_one_image(frame, bboxes)
        preds = self.instances_list_from_preds(preds)

        # return the prediction
        return preds

    def process_one_video(
        self,
        video_path,
        output_directory,
        bboxes_list=None,
        show_interval=0,
        callback=lambda x, y, z: None,
    ):
        """
        Apply pose estimation on one video and save the results.

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
        video_extension = "." + video_name.split(".")[-1]

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writter = cv2.VideoWriter(
            output_directory + "visualizations/" + video_name,
            fourcc,
            info_dict["fps"],
            (info_dict["width"], info_dict["height"]),
        )

        num_frames = 0
        target_resolution = (256, 256)

        start_time = time.time()

        # read video
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()

            frame_dict = {}
            frame_dict["frame_id"] = num_frames

            try:
                if bboxes_list is not None:
                    bboxes = bboxes_list[num_frames]
                else:
                    bboxes = None
                # topdown pose estimation
                preds = self.infer_one_image(
                    frame, bboxes=bboxes, show_interval=show_interval
                )
                frame_dict["instances"] = self.instances_list_from_preds(preds)
                img_output = self.visualizer.get_image()
            except KeyError as e:
                print(e)
                break

            num_frames += 1
            img_output = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
            video_writter.write(img_output)
            pose_sequence.append(frame_dict)

            fps = num_frames / (time.time() - start_time)
            callback(num_frames, fps, info_dict)

        with open(
            output_directory
            + "predictions/"
            + video_name.replace(video_extension, ".json"),
            "w",
        ) as f:
            json.dump(pose_sequence, f)


def main():

    pose_inferencer = PoseInferencer("RTMDetM", "RTMPoseM")

    args = parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_directory + "visualizations", exist_ok=True)
    os.makedirs(args.output_directory + "predictions", exist_ok=True)

    if args.image_path is not None:
        pose_inferencer.process_one_image(args.image_path, args.output_directory, None)

    elif args.video_path is not None:
        pose_inferencer.process_one_video(args.video_path, args.output_directory)

    elif args.video_directory is not None:
        for file_name in os.listdir(args.video_directory):
            file_path = os.path.join(args.video_directory, file_name)
            print("Processing : ", file_path)
            pose_inferencer.process_one_video(file_path, args.output_directory)
        print("---- Done ----")


if __name__ == "__main__":

    main()
