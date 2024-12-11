import matplotlib.pyplot as plt
import seaborn as sns
from angles_computation import *

def plot_elbow_angles(df):
         
        """
        Plots the angles.
    
        Args:
            df (dataframe): dataframe containing the angles for each pose
    
        Returns:
            fig (figure): figure containing the plots
            ax (axes): axes containing the plots
        """

        fig, ax = plt.subplots(figsize=(16, 8))

        sns.lineplot(x=df.index, y=df['left_elbow_angle'], ax=ax, label='left_elbow_angle')
        sns.lineplot(x=df.index, y=df['right_elbow_angle'], ax=ax, label='right_elbow_angle')
        ax.set_title('Elbow angles')
        ax.set_ylabel('Angle (°)')
        ax.set_xlabel('Time (frames)')
        ax.legend()

        return fig, ax

def plot_hip_angles(df):
             
    """
    Plots the angles.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots
        ax (axes): axes containing the plots
    """
    
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_hip_angle'], ax=ax, label='left_hip_angle')
    sns.lineplot(x=df.index, y=df['right_hip_angle'], ax=ax, label='right_hip_angle')
    ax.set_title('Hip angles')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax

def plot_knee_angles(df):
                     
    """
    Plots the angles.

    Args:
        df (dataframe): dataframe containing the angles for each pose

    Returns:
        fig (figure): figure containing the plots  
        ax (axes): axes containing the plots
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(x=df.index, y=df['left_knee_angle'], ax=ax, label='left_knee_angle')
    sns.lineplot(x=df.index, y=df['right_knee_angle'], ax=ax, label='right_knee_angle')
    ax.set_title('Knee angles')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (frames)')
    ax.legend()

    return fig, ax
