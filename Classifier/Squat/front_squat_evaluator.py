import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from squat_evaluator import SquatEvaluator
import matplotlib.pyplot as plt

class FrontSquatEvaluator(SquatEvaluator):
    
    def __init__(self, predictions_path, person_id=0):
        super().__init__(predictions_path, person_id)

        self.knees_outside_shoulders_condition = [True for i in range(len(self.squats_indexes))]
        self.shoulders_leveled_condition = [True for i in range(len(self.squats_indexes))]

        self.squat_correctness = [True for i in range(len(self.squats_indexes))]

    
    def check_knees_outside_shoulders(self, squat_number):
        """
        Check if the knees are outside the shoulders for a squat
        """

        condition_verified = True
        for frame in range(len(self.x_position_df_list[squat_number])):
            if self.x_position_df_list[squat_number].loc[frame, 'left_knee'] < self.x_position_df_list[squat_number].loc[frame, 'left_shoulder']:
                self.x_position_df_list[squat_number].loc[frame, 'correctness'] = False
                self.correctness_list[squat_number][frame] = False
                condition_verified = False
            if self.x_position_df_list[squat_number].loc[frame, 'right_knee'] > self.x_position_df_list[squat_number].loc[frame, 'right_shoulder']:
                self.x_position_df_list[squat_number].loc[frame, 'correctness'] = False
                self.correctness_list[squat_number][frame] = False
                condition_verified = False
        return condition_verified
    
    def check_shoulders_leveled(self, squat_number, threshold=0.04):
        """
        Check if the shoulders are leveled for one squat

        Args:
            squat_number (int): The number of the squat

        Returns:
            bool: True if the shoulders are leveled for one squat, False otherwise
        """
        y_position_left_shoulder = self.y_position_df_list[squat_number]['left_shoulder']
        y_position_right_shoulder = self.y_position_df_list[squat_number]['right_shoulder']

        squat_pose = self.squats_sequence[squat_number]
        bbox_height = squat_pose[0]['instances'][self.person_id]['bbox'][0][3] - squat_pose[0]['instances'][self.person_id]['bbox'][0][1]

        condition_verified = True

        for i in range(len(y_position_left_shoulder)):
            diff = abs(y_position_left_shoulder[i] - y_position_right_shoulder[i])
            if diff / bbox_height > threshold:
                self.correctness_list[squat_number][i] = False
                condition_verified = False
            
        return condition_verified


    def check_squat(self):

        for squat in range(len(self.squats_indexes)):
            # Check if knees are outside shoulders
            if not self.check_knees_outside_shoulders(squat):
                self.knees_outside_shoulders_condition[squat] = False
            # Check if shoulders are leveled
            if not self.check_shoulders_leveled(squat):
                self.shoulders_leveled_condition[squat] = False

            if not self.knees_outside_shoulders_condition[squat] or not self.shoulders_leveled_condition[squat]:
                self.squat_correctness[squat] = False



if __name__=='__main__':
    front_squat_evaluator = FrontSquatEvaluator('outputs/predictions/front_fail.json', 0)
    print(front_squat_evaluator.squats_indexes)

    front_squat_evaluator.check_squat()
    print(front_squat_evaluator.squat_correctness)

    print(front_squat_evaluator.shoulders_leveled_condition)
    print(front_squat_evaluator.knees_outside_shoulders_condition)

    print(front_squat_evaluator.correctness_list[1])
