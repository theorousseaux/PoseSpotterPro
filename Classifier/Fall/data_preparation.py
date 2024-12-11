import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split 


class WindowFeatureExtractor():

    def __init__(self, window_size=10, step_size=5):
        self.window_size = window_size
        self.step_size = step_size
        self.is_interactive = sys.stdout.isatty()

        self.id_joints_dict = {0: 'nose',
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
        self.joints_id_dict = {v: k for k, v in self.id_joints_dict.items()}

    
    def get_df_from_preds(self, preds, instance_id=0):

        """
        Get a dataframe from a json file containing the poses. The dataframe contains the coordinates of the joints of the instance_id-th.

        Args:
            preds (str or list): Path of the json file containing the poses or list of poses.

        Returns:
            df (pd.DataFrame): A dataframe of poses.
        """

        if type(preds) == list:
            pose_sequence = preds
        else:
            with open(preds) as json_file:
                pose_sequence = json.load(json_file)
        
        dic_list = []

        for i in range(len(pose_sequence)):
            keypoints_dict = {}

            try:
                keypoints_list = pose_sequence[i]['instances'][instance_id]['keypoints']
            except:
                keypoints_list = [[np.nan, np.nan] for i in range(17)]

            for number, keypoint in enumerate(keypoints_list):
                keypoints_dict["X_" + self.id_joints_dict[number]] = keypoint[0]
                keypoints_dict["Y_" + self.id_joints_dict[number]] = keypoint[1]
            
            dic_list.append(keypoints_dict)
            
        df = pd.DataFrame.from_dict(dic_list)
        return df


    def get_sliding_windows(self, df):
        """
        Returns a list of dataframes of size window_size with step_size between each dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe to be split into windows.
            window_size (int): Size of each window.
            step_size (int): Number of rows to skip between each window.
            
        Returns:
            windows (list): List of dataframes of size window_size with step_size between each dataframe.
        """
        
        windows = []
        for i in range(0, len(df) - self.window_size + 1, self.step_size):
            windows.append(df[i:i+self.window_size])
    
        return windows
    
    def get_features_from_window(self, window):
        """
        Extract features from a window of data. 

        Args:
            window (np.array): A window of data

        Returns:
            features (dict): A dictionary of features
        """
        features = {}

        for keypoint in window.columns:

            if keypoint == 'label':
                continue

            features[keypoint + '_mean'] = window[keypoint].mean()
            features[keypoint + '_std'] = window[keypoint].std()
            features[keypoint + '_min'] = window[keypoint].min()
            features[keypoint + '_max'] = window[keypoint].max()
                
        return features

    
    def create_dataset(self, windows):
        """
        Create a dataset from a list of windows.

        Args:
            windows (list): A list of windows

        Returns:
            X (pd.DataFrame): A dataframe of features
            y (np.array): An array of labels
        """

        features = []
        window_labels = []
        for window in tqdm(windows, desc='Creating dataset from windows', disable = not self.is_interactive, colour='green'):
            features.append(self.get_features_from_window(window))
            if 'fall' in window['label'].unique():
                window_labels.append('Fall')
            else:
                window_labels.append(window['label'].iloc[-1])

        return pd.DataFrame(features), np.array(window_labels)
    

    def create_sample(self, df):

        """
        Get features from a dataframe. Returns a dataframe ready to be used for an inference.

        Args:
            df (pd.DataFrame): A dataframe of poses

        Returns:
            features_df (pd.DataFrame): A dataframe of features
        """

        features = []
        windows = self.get_sliding_windows(df)
        for window in tqdm(windows, desc='Creating dataset from windows', disable = not self.is_interactive, colour='green'):
            features.append(self.get_features_from_window(window))

        return pd.DataFrame(features)


    def prepare_data(self, files_list):

        """
        Prepare data for training and testing.

        Args:
            files_list (list): A list containing the paths of the files to be used for training/testing.

        Returns:    
            X (pd.DataFrame): A dataframe of features
            y (np.array): An array of labels
        """

        
        files_list.sort()
        windows = []

        for file_index in tqdm(range(len(files_list)), desc='Creating windows', disable = not self.is_interactive):
            df = pd.read_csv(files_list[file_index], index_col=0)
            new_windows = self.get_sliding_windows(df)
            windows += new_windows

        X, y = self.create_dataset(windows)

        return X, y

if __name__ == '__main__':

    poses_sequence_path = 'Data/Fall/Dataset_CAUCAFall/Poses_sequences/'
    tot_file_list = os.listdir(poses_sequence_path)
    tot_file_list = [poses_sequence_path + file_name for file_name in tot_file_list]


    window_size = 10
    step_size = 5

    window_feature_extractor = WindowFeatureExtractor(window_size, step_size)
    X, y = window_feature_extractor.prepare_data(tot_file_list)

    print(X.head())
    print('y shape:', y.shape)

    # Check labels distribution
    print(np.unique(y, return_counts=True))
