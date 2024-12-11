import json
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from FeaturesExtractor.Angles.angles_computation import compute_angles, list_to_dataframe
import matplotlib.pyplot as plt
import numpy as np

id_joints_dict = {0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'}
joints_id_dict = {v: k for k, v in id_joints_dict.items()}

class SquatEvaluator:

    def __init__(self, prediction_path, person_id=0):

        self.id_joints_dict = id_joints_dict
        self.joints_id_dict = joints_id_dict

        with open(prediction_path) as f:
            self.pose_sequence = json.load(f)
        self.person_id = person_id
        self.angles_df = list_to_dataframe(compute_angles(self.pose_sequence), person_id=self.person_id)

        self.squats_indexes = self.find_squats_indexes(0.2)
        self.squats_sequence = self.compute_squats_sequence(self.pose_sequence, self.squats_indexes)  

        self.x_position_df_list = self.compute_x_position_df_list()
        self.y_position_df_list = self.compute_y_position_df_list()

        self.correctness_list = []

        for i in range(len(self.squats_indexes)):
            self.correctness_list.append([True for j in range(len(self.x_position_df_list[i]))])


    def find_squats_indexes(self, threshold=0.2):
        
        y_position_nose = self.normalize_y_position('nose')

        squats_indexes = []
        squat_start = None

        for i in range(1, len(y_position_nose)):
            if squat_start is None and y_position_nose[i] > threshold:
                squat_start = i
            elif squat_start is not None and y_position_nose[i] < threshold:
                squats_indexes.append((squat_start, i))
                squat_start = None
        
        return squats_indexes

    
    def compute_squats_sequence(self, pose_sequence, squats_indexes):
        squats_sequence = []
        for i, (start, end) in enumerate(squats_indexes):
            pose = pose_sequence[start:end]
            squats_sequence.append(pose)
        return squats_sequence
    
    
    def compute_x_position(self, squat_number, joint):
        """
        Compute the x position of a joint for one squat

        Args:
            squat_number (int): The number of the squat
            joint (str): The joint to compute the x position

        Returns:
            list: The x position of the joint for one squat
        """

        x_position = []
        squat_pose = self.squats_sequence[squat_number]

        for pose in squat_pose:
            x_position.append(pose['instances'][self.person_id]['keypoints'][self.joints_id_dict[joint]][0])

        return x_position
    
    def compute_x_position_df_list(self):
        """
        Compute the x position of all the joints for all the squats

        Returns:
            list: A list of dataframe containing the x position of all the joints for all the squats
        """
            
        x_position_df_list = []

        for i in range(len(self.squats_indexes)):
            dict = {}
            for joint in self.joints_id_dict.keys():
                dict[joint] = self.compute_x_position(i, joint)
            position_df = pd.DataFrame(dict)
            position_df['correctness'] = [True for i in range(len(position_df))]
            x_position_df_list.append(position_df)

        return x_position_df_list

    def compute_y_position(self, squat_number, joint):
        """
        Compute the y position of a joint for one squat

        Args:
            squat_number (int): The number of the squat
            joint (str): The joint to compute the y position

        Returns:
            list: The y position of the joint for one squat
        """

        y_position = []
        squat_pose = self.squats_sequence[squat_number]

        for pose in squat_pose:
            y_position.append(pose['instances'][self.person_id]['keypoints'][self.joints_id_dict[joint]][1])

        return y_position
    
    def compute_y_position_df_list(self):
        """
        Compute the y position of all the joints for all the squats

        Returns:
            list: A list of dataframe containing the y position of all the joints for all the squats
        """
            
        y_position_df_list = []

        for i in range(len(self.squats_indexes)):
            dict = {}
            for joint in self.joints_id_dict.keys():
                dict[joint] = self.compute_y_position(i, joint)
            position_df = pd.DataFrame(dict)
            position_df['correctness'] = [True for i in range(len(position_df))]
            y_position_df_list.append(position_df)

        return y_position_df_list
    
    
    def normalize_y_position(self, joint):
        """
        Normalize the y position of a joint for the whole sequence

        Args:
            joint (str): The joint to normalize the y position

        Returns:
            list: The normalized y position of the joint for the whole sequence
        """
        y_position = []
        for pose in self.pose_sequence:
            y_position.append(pose['instances'][self.person_id]['keypoints'][self.joints_id_dict[joint]][1])

        return [(y - min(y_position)) / (max(y_position) - min(y_position)) for y in y_position]
    
    
    def check_squat(self):
        pass

    def error_frame_index(self, squat_number):
        """
        Return the index of the first frame where the squat is not correct

        Args:
            squat_number (int): The number of the squat

        Returns:
            tuple: The index of the first frame where the squat is not correct and the end of the incorrect sequence
        """
        start = None
        end = None
        for i in range(len(self.correctness_list[squat_number])):
            if not self.correctness_list[squat_number][i]:
                start = i
                break
        for i in range(start, len(self.correctness_list[squat_number])):
            if self.correctness_list[squat_number][i]:
                end = i
                break
            else:
                end = len(self.correctness_list[squat_number])
        return (start, end)



if __name__ == '__main__':
    evaluator = SquatEvaluator('UserInterface/outputs/predictions/front_fail.json', person_id=0)
    print(evaluator.y_position_df_list[0].head())


    import numpy as np

    joint = 'nose'
    y_position_tot = evaluator.normalize_y_position(joint)
    threshold = 0.2

    # Tracer la courbe
    plt.plot(y_position_tot)

    # Tracer la ligne horizontale à y=threshold
    plt.axhline(y=threshold, color='r', linestyle='--')

    # Annoter le début et la fin du squat
    for i in range(1, len(y_position_tot)):
        if y_position_tot[i-1] <= threshold and y_position_tot[i] > threshold:
            plt.annotate('Squat Start', (i, threshold), xytext=(i, threshold+0.05), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)
        elif y_position_tot[i-1] > threshold and y_position_tot[i] <= threshold:
            plt.annotate('Squat End', (i, threshold), xytext=(i, threshold-0.1), arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)

    # Annoter l'axe vertical
    plt.annotate('Top of the frame', (0, 0), xytext=(-0.07, 0), textcoords='axes fraction', va='bottom', ha='right', fontsize=9, rotation=90)
    plt.annotate('Bottom of the frame', (0, 1), xytext=(-0.07, 1), textcoords='axes fraction', va='top', ha='right', fontsize=9, rotation=90)


    # Autres éléments du graphique
    plt.xlabel('Frame')
    plt.ylabel('Normalized y position', labelpad=20)
    plt.title('Normalized y position of the ' + joint + ' joint' + ' during the whole sequence')
    plt.tight_layout()
    plt.savefig('y_position_{}_tot_normalized.png'.format(joint))
