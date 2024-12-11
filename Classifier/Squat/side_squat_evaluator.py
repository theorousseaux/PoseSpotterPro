import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from squat_evaluator import SquatEvaluator
import matplotlib.pyplot as plt

class SideSquatEvaluator(SquatEvaluator):

    def __init__(self, predictions_path, side, person_id):
        super().__init__(predictions_path, person_id)

        self.angles_df = self.compute_squat_id(self.angles_df, self.squats_indexes)
        self.side = side

        # Conditions to check
        self.knee_angle_condition = [True for i in range(len(self.squats_indexes))]
        self.shoulder_behind_ankle_condition = [True for i in range(len(self.squats_indexes))]
        self.squat_correctness = [True for i in range(len(self.squats_indexes))]


    def compute_squat_id(self, angles_df, squats_indexes):
        """
        Compute the squat id for each row of the angles dataframe

        Args:
            angles_df (pandas.DataFrame): The angles dataframe
            squats_indexes (list): The list of squats indexes (start, end)

        Returns:
            pandas.DataFrame: The angles dataframe with the squat id
        """

        angles_df['squat_id'] = -1  # initialize the column with -1

        for i, (start, end) in enumerate(squats_indexes):
            self.angles_df.loc[start:end, 'squat_id'] = i  # assign the squat id to the corresponding rows
        
        return angles_df
    
    
    def check_knee_angle(self, squat_df, min_knee_angle_threshold, knee_angle='left_knee_angle'):
        """
        Check if the knee angle is between the thresholds for one squat

        Args:
            squat_df (pandas.DataFrame): The dataframe containing the angles for one squat
            min_knee_angle_threshold (int): The minimum knee angle to accept the squat
            knee_angle (str): The knee angle to check

        Returns:
            bool: True if the knee angle is between the thresholds for one squat, False otherwise
        """
            
        return (squat_df[knee_angle] < min_knee_angle_threshold).any()

    def check_shoulder_position(self, squat_number, tolerance_factor=0.09):

        """
        Check if the shoulder is behind the toe for one squat

        Args:
            squat_number (int): The number of the squat
            tolerance_factor (float): The tolerance factor to accept the squat, it represents the percentage of the bbox height

        Returns:
            bool: True if the shoulder is behind the toe for one squat, False otherwise
        """

        tolerance_list = []
        shoulder_behind_ankle = []
        squat_pose = self.squats_sequence[squat_number]

        bbox_height = squat_pose[0]['instances'][self.person_id]['bbox'][0][3] - squat_pose[0]['instances'][self.person_id]['bbox'][0][1]

        if self.side == 'left':
            shoulder = 'left_shoulder'
            ankle = 'left_ankle'
        else:
            shoulder = 'right_shoulder'
            ankle = 'right_ankle'

        for (i,pose) in enumerate(squat_pose):
            tolerance = tolerance_factor * bbox_height
            tolerance_list.append(tolerance)

            if self.side == 'left':
                condition = self.x_position_df_list[squat_number][shoulder][i] > self.x_position_df_list[squat_number][ankle][i] - tolerance
                self.correctness_list[squat_number][i] = condition
                shoulder_behind_ankle.append(condition)
            else:
                condition = self.x_position_df_list[squat_number][shoulder][i] < self.x_position_df_list[squat_number][ankle][i] + tolerance
                self.correctness_list[squat_number][i] = condition
                shoulder_behind_ankle.append(condition)

        return shoulder_behind_ankle, self.x_position_df_list[squat_number][shoulder], self.x_position_df_list[squat_number][ankle], tolerance_list
    
    def plot_shoulder_ankle(self, squat_number, path=""):
        _, shoulder_x_list, ankle_x_list, tolerance_list = self.check_shoulder_position(squat_number)
        
        plt.plot(shoulder_x_list, 'r', label='shoulder')
        plt.plot(ankle_x_list, 'b', label='ankle')
        
        lower_tolerance = [a - t for a, t in zip(ankle_x_list, tolerance_list)]
        upper_tolerance = [a + t for a, t in zip(ankle_x_list, tolerance_list)]
        
        plt.fill_between(range(len(ankle_x_list)), lower_tolerance, upper_tolerance, color='b', alpha=0.1, label='tolerance')

        pos_mean = (sum(shoulder_x_list) + sum(ankle_x_list)) / (len(shoulder_x_list) + len(ankle_x_list))
        
        plt.ylim(pos_mean - 400, pos_mean + 400)
        plt.gca().invert_yaxis()
        plt.xlabel('Frame')
        plt.ylabel('X position')
        plt.title('Shoulder and ankle horizontal position over time')
        plt.legend()

        plt.savefig(path + 'shoulder_ankle_{}_side_{}.png'.format(self.side, squat_number+1))

    
    def check_squat(self):
        """
        Check if the squats are correct
        """
        filtered_angles_df = self.angles_df[self.angles_df['squat_id'] != -1]

        for squat_id, squat_df in filtered_angles_df.groupby('squat_id'):

            knee_angle = 'left_knee_angle' if self.side == 'left' else 'right_knee_angle'
            self.knee_angle_condition[squat_id] = self.check_knee_angle(squat_df, 90, knee_angle=knee_angle)

            shoulder_behind_ankle, _, _, _ = self.check_shoulder_position(squat_id)
            if False in shoulder_behind_ankle:
                self.shoulder_behind_ankle_condition[squat_id] = False
        
        for i in range(len(self.knee_angle_condition)):
            self.squat_correctness[i] = self.knee_angle_condition[i] and self.shoulder_behind_ankle_condition[i]


if __name__ == '__main__':
    evaluator = SideSquatEvaluator('outputs/predictions/shoulder_fail.json', person_id=0, side='right')
    print(evaluator.angles_df['squat_id'].unique())
    evaluator.check_squat()
    print("knee angle condition :", evaluator.knee_angle_condition)
    print("shoulder position condition :", evaluator.shoulder_behind_ankle_condition)
    print(evaluator.error_frame_index(1))