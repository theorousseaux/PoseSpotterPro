import matplotlib.pyplot as plt
import seaborn as sns
import json
from angles_computation import *

def plot_right_elbow_angle(df):
         
    """
    Plots the evolution of the right elbow angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['right_elbow_angle'], ax=ax, label='right_elbow_angle')
    ax.set_title('right_elbow_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_left_elbow_angle(df):
             
    """
    Plots the evolution of the left elbow angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_elbow_angle'], ax=ax, label='left_elbow_angle')
    ax.set_title('left_elbow_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_right_hip_angle(df):

    """
    Plots the evolution of the right hip angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['right_hip_angle'], ax=ax, label='right_hip_angle')
    ax.set_title('right_hip_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_left_hip_angle(df):

    """
    Plots the evolution of the left hip angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_hip_angle'], ax=ax, label='left_hip_angle')
    ax.set_title('left_hip_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_right_knee_angle(df):

    """
    Plots the evolution of the right knee angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['right_knee_angle'], ax=ax, label='right_knee_angle')
    ax.set_title('right_knee_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_left_knee_angle(df):

    """
    Plots the evolution of the left knee angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_knee_angle'], ax=ax, label='left_knee_angle')
    ax.set_title('left_knee_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_right_shoulder_angle(df):

    """
    Plots the evolution of the right shoulder angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['right_shoulder_angle'], ax=ax, label='right_shoulder_angle')
    ax.set_title('right_shoulder_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_left_shoulder_angle(df):

    """
    Plots the evolution of the left shoulder angle.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_shoulder_angle'], ax=ax, label='left_shoulder_angle')
    ax.set_title('left_shoulder_angle')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

if __name__ == "__main__":
    with open('outputs/predictions/tibs.json') as json_file:
        poses_sequences = json.load(json_file) 

    angles_list = compute_angles(poses_sequences)
    df = list_to_dataframe(angles_list)

    fig, ax = plot_right_elbow_angle(df)
    plt.savefig('outputs/plots/right_elbow_angle.png')

    fig, ax = plot_left_elbow_angle(df)
    plt.savefig('outputs/plots/left_elbow_angle.png')

    fig, ax = plot_right_hip_angle(df)
    plt.savefig('outputs/plots/right_hip_angle.png')

    fig, ax = plot_left_hip_angle(df)
    plt.savefig('outputs/plots/left_hip_angle.png')

    fig, ax = plot_right_knee_angle(df)
    plt.savefig('outputs/plots/right_knee_angle.png')

    fig, ax = plot_left_knee_angle(df)
    plt.savefig('outputs/plots/left_knee_angle.png')

    fig, ax = plot_right_shoulder_angle(df)
    plt.savefig('outputs/plots/right_shoulder_angle.png')

    fig, ax = plot_left_shoulder_angle(df)
    plt.savefig('outputs/plots/left_shoulder_angle.png')

