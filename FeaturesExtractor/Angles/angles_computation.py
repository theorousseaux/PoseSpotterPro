import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from single_angle_plot import *

#################### GLOABAL VARIABLES ####################

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

############################################################



def compute_left_knee_angle(pose):

    """
    Computes the knee angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        knee_angle (float): knee angle
    """

    left_hip = pose['keypoints'][joints_id_dict['left_hip']]
    left_knee = pose['keypoints'][joints_id_dict['left_knee']]
    left_ankle = pose['keypoints'][joints_id_dict['left_ankle']]

    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    return knee_angle

def compute_right_knee_angle(pose):

    """
    Computes the knee angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        knee_angle (float): knee angle
    """

    right_hip = pose['keypoints'][joints_id_dict['right_hip']]
    right_knee = pose['keypoints'][joints_id_dict['right_knee']]
    right_ankle = pose['keypoints'][joints_id_dict['right_ankle']]

    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    return knee_angle

def compute_left_hip_angle(pose):

    """
    Computes the left hip angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        hip_angle (float): hip angle
    """

    left_shoulder = pose['keypoints'][joints_id_dict['left_shoulder']]
    left_hip = pose['keypoints'][joints_id_dict['left_hip']]
    left_knee = pose['keypoints'][joints_id_dict['left_knee']]

    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    return hip_angle

def compute_right_hip_angle(pose):
    
    """
    Computes the right hip angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        hip_angle (float): hip angle
    """

    right_shoulder = pose['keypoints'][joints_id_dict['right_shoulder']]
    right_hip = pose['keypoints'][joints_id_dict['right_hip']]
    right_knee = pose['keypoints'][joints_id_dict['right_knee']]

    hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    return hip_angle

def compute_left_elbow_angle(pose):
    
    """
    Computes the left elbow angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        elbow_angle (float): elbow angle
    """

    left_shoulder = pose['keypoints'][joints_id_dict['left_shoulder']]
    left_elbow = pose['keypoints'][joints_id_dict['left_elbow']]
    left_wrist = pose['keypoints'][joints_id_dict['left_wrist']]

    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    return elbow_angle

def compute_right_elbow_angle(pose):
        
    """
    Computes the right elbow angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        elbow_angle (float): elbow angle
    """

    right_shoulder = pose['keypoints'][joints_id_dict['right_shoulder']]
    right_elbow = pose['keypoints'][joints_id_dict['right_elbow']]
    right_wrist = pose['keypoints'][joints_id_dict['right_wrist']]

    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return elbow_angle

def compute_left_shoulder_angle(pose):
            
    """
    Computes the left shoulder angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        shoulder_angle (float): shoulder angle
    """

    left_hip = pose['keypoints'][joints_id_dict['left_hip']]
    left_shoulder = pose['keypoints'][joints_id_dict['left_shoulder']]
    left_elbow = pose['keypoints'][joints_id_dict['left_elbow']]

    shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

    return shoulder_angle

def compute_right_shoulder_angle(pose):
                    
    """
    Computes the right shoulder angle.

    Args:
        pose (dict): pose dictionary

    Returns:
        shoulder_angle (float): shoulder angle
    """

    right_hip = pose['keypoints'][joints_id_dict['right_hip']]
    right_shoulder = pose['keypoints'][joints_id_dict['right_shoulder']]
    right_elbow = pose['keypoints'][joints_id_dict['right_elbow']]

    shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

    return shoulder_angle   


def compute_angles(poses_sequence):

    """
    Computes the angles for each pose in the pose sequences.

    Args:
        pose_sequences (list): list of pose sequences returned by the pose estimation model

    Returns:
        angles_list (list): list of dictionaries containing the angles for each pose
    """

    angles_list = []

    for frame in poses_sequence:

        angles_dict = {}
        angles_dict['frame_id'] = frame['frame_id']
        angles_dict['instances'] = []

        for instance in frame['instances']:

            left_knee_angle = compute_left_knee_angle(instance)
            right_knee_angle = compute_right_knee_angle(instance)
            left_hip_angle = compute_left_hip_angle(instance)
            right_hip_angle = compute_right_hip_angle(instance)
            left_elbow_angle = compute_left_elbow_angle(instance)
            right_elbow_angle = compute_right_elbow_angle(instance)
            left_shoulder_angle = compute_left_shoulder_angle(instance)
            right_shoulder_angle = compute_right_shoulder_angle(instance)

            angles_dict['instances'].append({'left_knee_angle': left_knee_angle,
                                    'right_knee_angle': right_knee_angle,
                                    'left_hip_angle': left_hip_angle,
                                    'right_hip_angle': right_hip_angle,
                                    'left_elbow_angle': left_elbow_angle,
                                    'right_elbow_angle': right_elbow_angle,
                                    'left_shoulder_angle': left_shoulder_angle,
                                    'right_shoulder_angle': right_shoulder_angle})
        
        angles_list.append(angles_dict)

    return angles_list

def list_to_dataframe(angles_list, person_id):

    """
    Converts the list of dictionaries containing the angles for each pose of a person to a dataframe.

    Args:
        angles_list (list): list of dictionaries containing the angles for each pose
        person_id (int): person id of the person whose angles are being converted to a dataframe

    Returns:
        df (dataframe): dataframe containing the angles for each pose
    """
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    left_elbow_angles = []
    right_elbow_angles = []
    left_soulder_angles = []
    right_soulder_angles = []

    for frame in angles_list:
        left_knee_angles.append(frame['instances'][person_id]['left_knee_angle'])
        right_knee_angles.append(frame['instances'][person_id]['right_knee_angle'])
        left_hip_angles.append(frame['instances'][person_id]['left_hip_angle'])
        right_hip_angles.append(frame['instances'][person_id]['right_hip_angle'])
        left_elbow_angles.append(frame['instances'][person_id]['left_elbow_angle'])
        right_elbow_angles.append(frame['instances'][person_id]['right_elbow_angle'])
        left_soulder_angles.append(frame['instances'][person_id]['left_shoulder_angle'])
        right_soulder_angles.append(frame['instances'][person_id]['right_shoulder_angle'])

    df = pd.DataFrame({'left_knee_angle': left_knee_angles,
                        'right_knee_angle': right_knee_angles,
                        'left_hip_angle': left_hip_angles,
                        'right_hip_angle': right_hip_angles,
                        'left_elbow_angle': left_elbow_angles,
                        'right_elbow_angle': right_elbow_angles,
                        'left_shoulder_angle': left_soulder_angles,
                        'right_shoulder_angle': right_soulder_angles})
    
    return df

def save_all_single_angle_plots(df, path='outputs/plots/'):
        
    """
    Saves all the plots in the given path.

    Args:
        df (dataframe): dataframe containing the angles for each pose
        path (str): path where to save the plots
    """

    plot_right_elbow_angle(df)[0].savefig(path + 'right_elbow_angle.png')
    plot_left_elbow_angle(df)[0].savefig(path + 'left_elbow_angle.png')
    plot_right_shoulder_angle(df)[0].savefig(path + 'right_shoulder_angle.png')
    plot_left_shoulder_angle(df)[0].savefig(path + 'left_shoulder_angle.png')
    plot_right_hip_angle(df)[0].savefig(path + 'right_hip_angle.png')
    plot_left_hip_angle(df)[0].savefig(path + 'left_hip_angle.png')
    plot_right_knee_angle(df)[0].savefig(path + 'right_knee_angle.png')
    plot_left_knee_angle(df)[0].savefig(path + 'left_knee_angle.png')

     


def plot_angles(df):

    """
    Plots the angles.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(3, 1, figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_knee_angle'], ax=ax[0], label='left_knee_angle')
    sns.lineplot(x=df.index, y=df['right_knee_angle'], ax=ax[0], label='right_knee_angle')
    ax[0].set_title('Knee angles')
    ax[0].set_ylabel('Angle (°)')
    ax[0].set_xlabel('Time (frames)')

    sns.lineplot(x=df.index, y=df['left_hip_angle'], ax=ax[1], label='left_hip_angle')
    sns.lineplot(x=df.index, y=df['right_hip_angle'], ax=ax[1], label='right_hip_angle')
    ax[1].set_title('Hip angles')
    ax[1].set_ylabel('Angle (°)')
    ax[1].set_xlabel('Time (frames)')

    sns.lineplot(x=df.index, y=df['left_elbow_angle'], ax=ax[2], label='left_elbow_angle')
    sns.lineplot(x=df.index, y=df['right_elbow_angle'], ax=ax[2], label='right_elbow_angle')
    ax[2].set_title('Elbow angles')
    ax[2].set_ylabel('Angle (°)')
    ax[2].set_xlabel('Time (frames)')

    plt.subplots_adjust(hspace = 0.4)

    return fig, ax

if __name__ == '__main__':
    
    with open('outputs/predictions/tibs.json') as json_file:
        poses_sequences = json.load(json_file) 

    angles_list = compute_angles(poses_sequences)
    df = list_to_dataframe(angles_list, person_id=0)

    fig, ax = plot_angles(df)
    plt.savefig('outputs/visualizations/all_angles.png')
    plt.close()