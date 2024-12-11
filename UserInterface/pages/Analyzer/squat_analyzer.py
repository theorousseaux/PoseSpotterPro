import streamlit as st
import altair as alt
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from Classifier.Squat import side_squat_evaluator, front_squat_evaluator
from MediaLoader.utils import *

def plot_x_position(evaluator, x_position_to_plot, squat_number):
    df = evaluator.x_position_df_list[squat_number-1][x_position_to_plot + ['correctness']]
    df.reset_index(inplace=True, names='frame')
    df_melted = pd.melt(df, id_vars=['frame', 'correctness'], value_vars=x_position_to_plot, var_name='joint', value_name='Horizontal position')
    x_position_chart = alt.Chart(data=df_melted, title='Evolution of the horizontal position during time').mark_line().encode(
        x=alt.X('frame', axis=alt.Axis(title='Frame')),
        y='Horizontal position',
        color='joint',
        tooltip=['frame', 'Horizontal position', 'correctness']
    ).interactive()
    st.altair_chart(x_position_chart, use_container_width=True)

def plot_y_position(evaluator, y_position_to_plot, squat_number):
    df = evaluator.y_position_df_list[squat_number-1][y_position_to_plot + ['correctness']]
    df.reset_index(inplace=True, names='frame')
    df_melted = pd.melt(df, id_vars=['frame', 'correctness'], value_vars=y_position_to_plot, var_name='joint', value_name='Vertical position')
    y_position_chart = alt.Chart(data=df_melted, title='Evolution of the vertical position during time').mark_line().encode(
        x=alt.X('frame', axis=alt.Axis(title='Frame')),
        y='Vertical position',
        color='joint',
        tooltip=['frame', 'Vertical position', 'correctness']
    ).interactive()
    st.altair_chart(y_position_chart, use_container_width=True)

def plot_angles(evaluator, angles_to_plot, squat_number):
    angles_df = evaluator.angles_df[angles_to_plot]
    angles_df.reset_index(inplace=True, names='frame')
    angles_df = angles_df[:][evaluator.squats_indexes[squat_number-1][0]:evaluator.squats_indexes[squat_number-1][1]]
    df_melted = pd.melt(angles_df, id_vars=['frame'], value_vars=angles_to_plot, var_name='joint', value_name='Angle value')
    angles_chart = alt.Chart(data=df_melted, title='Evolution of the angles during time').mark_line().encode(
        x=alt.X('frame', axis=alt.Axis(title='Frame')),
        y=alt.Y('Angle value:Q', axis=alt.Axis(title='Angle value')),
        color='joint',
        tooltip=['frame', 'Angle value']
    ).interactive()
    st.altair_chart(angles_chart, use_container_width=True)

def plot_shoulder_ankle(evaluator, squat_number):
    _, shoulder_x_list, ankle_x_list, tolerance_list = evaluator.check_shoulder_position(squat_number-1)
    df = pd.DataFrame({'Shoulder x position': shoulder_x_list, 'Ankle x position': ankle_x_list, 'Tolerance': tolerance_list})
    df.reset_index(inplace=True, names='frame')
    df['tolerance_min'] = df['Ankle x position'] - df['Tolerance']
    df['tolerance_max'] = df['Ankle x position'] + df['Tolerance']

    df_melted = pd.melt(df, id_vars=['frame'], value_vars=['Shoulder x position', 'Ankle x position'], var_name='joint', value_name='Horizontal position')

    shoulder_ankle_chart = alt.Chart(data=df_melted, title='Evolution of the horizontal position during time').mark_line().encode(
        x=alt.X('frame', axis=alt.Axis(title='Frame')),
        y='Horizontal position',
        color='joint',
        tooltip=['frame', 'Horizontal position']
    ).interactive()
    area_chart = alt.Chart(data=df).mark_area(opacity=0.3).encode(
        x=alt.X('frame', axis=alt.Axis(title='Frame')),
        y=alt.Y('tolerance_min', title='Tolerance', axis=alt.Axis(labels=False)),
        y2='tolerance_max',
    )
    st.altair_chart(shoulder_ankle_chart + area_chart, use_container_width=True)


def side_squat_analyzer(json_file_path, person_id, point_of_view):
    """
    Streamlit page to analyze a side squat

    Args:
        json_file_path (str): path to the json file to analyze
        person_id (int): id of the person to analyze
        point_of_view (str): point of view of the video

    Returns:
        None
    """

    side_evaluator = side_squat_evaluator.SideSquatEvaluator(json_file_path, person_id=person_id, side='left' if point_of_view == 'Left side' else 'right')
    side_evaluator.check_squat()

    angles_df = side_evaluator.angles_df
    st.write('Angles computed for person {} detected on the first image on the file: {}'.format(person_id, json_file_path.split(os.sep)[-1]))

    selected_angles = st.multiselect('Select the angles to plot', side_evaluator.angles_df.columns)
    if selected_angles:
        st.line_chart(angles_df[selected_angles])

    
    st.markdown('## Squat correctness')
    for squat_id in range(len(side_evaluator.squat_correctness)):
        st.markdown('- Squat {} position: {}'.format(squat_id+1, ':white_check_mark:' if side_evaluator.squat_correctness[squat_id] else ':x:'))

    squat_number = st.selectbox('Choose a squat number', [i+1 for i in range(len(side_evaluator.squats_sequence))])

    if squat_number:
        angles_squat_df = angles_df[:][side_evaluator.squats_indexes[squat_number-1][0]:side_evaluator.squats_indexes[squat_number-1][1]]
        if not side_evaluator.squat_correctness[squat_number-1]:
            st.markdown('## Squat errors')
                
            if not side_evaluator.knee_angle_condition[squat_number-1]:
                st.markdown('**Knee angle**: :x:')

                st.markdown('    - Left knee angle minimum: {}°'.format(round(angles_squat_df['left_knee_angle'].min())))
                st.markdown('    - Right knee angle minimum: {}°'.format(round(angles_squat_df['right_knee_angle'].min())))
                st.markdown('You should try to reach a minimum of 90° for both knees.')
            
            if not side_evaluator.shoulder_behind_ankle_condition[squat_number-1]:
                st.markdown('**Shoulder behind ankle**: :x:')
                st.markdown('You should try to keep your shoulders in the same vertical line as your feet during the whole movement.')

                start, end = side_evaluator.error_frame_index(squat_number-1)
                start_frame = side_evaluator.squats_indexes[squat_number-1][0] + start
                end_frame = side_evaluator.squats_indexes[squat_number-1][0] + end
                st.markdown('**Error frames**: {} to {}'.format(start_frame, end_frame))
                os.makedirs('UserInterface/outputs/visualizations/error', exist_ok=True)
                error_path = cut_video_moviepy(st.session_state.visulization_path, start_frame, end_frame, 'UserInterface/outputs/visualizations/error/error.mp4')
                col1, _ = st.columns(2)
                with col1:
                    st.video(error_path)
        else:
            st.markdown('## Squat errors')
            st.markdown('**No errors detected**: :white_check_mark:')

        angles_to_plot = st.multiselect('Select the angles to plot', side_evaluator.angles_df.columns, default=['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle'])
        if angles_to_plot:
            plot_angles(side_evaluator, angles_to_plot, squat_number)
        plot_shoulder_ankle(side_evaluator, squat_number)


def front_squat_analyzer(json_file_path, person_id):

    front_evaluator = front_squat_evaluator.FrontSquatEvaluator(json_file_path, person_id=person_id)
    front_evaluator.check_squat()

    angles_df = front_evaluator.angles_df
    st.write('Angles computed for person {} detected on the first image on the file: {}'.format(person_id, json_file_path.split(os.sep)[-1]))

    selected_angles = st.multiselect('Select the angles to plot', front_evaluator.angles_df.columns)
    if selected_angles:
        st.line_chart(angles_df[selected_angles])


    st.markdown('## Squat correctness')

    for squat_id in range(len(front_evaluator.squat_correctness)):
        st.markdown('- Squat {} position: {}'.format(squat_id+1, ':white_check_mark:' if front_evaluator.squat_correctness[squat_id] else ':x:'))
    
    squat_number = st.selectbox('Choose a squat number', [i+1 for i in range(len(front_evaluator.squats_sequence))])

    if squat_number:
        if not front_evaluator.squat_correctness[squat_number-1]:
            st.markdown('## Squat errors')

            if not front_evaluator.knees_outside_shoulders_condition[squat_number-1]:
                st.markdown('**Knees outside shoulders**: :x:')
                st.markdown('You should try to keep your knees outside your shoulder lines during the whole movement.')
            
            if not front_evaluator.shoulders_leveled_condition[squat_number-1]:
                st.markdown('**Shoulders leveled**: :x:')
                st.markdown('You should try to keep your shoulders at the same level during the whole movement.')

            start, end = front_evaluator.error_frame_index(squat_number-1)
            start_frame = front_evaluator.squats_indexes[squat_number-1][0] + start
            end_frame = front_evaluator.squats_indexes[squat_number-1][0] + end
            st.markdown('**Error frames**: {} to {}'.format(start_frame, end_frame))
            os.makedirs('UserInterface/outputs/visualizations/error', exist_ok=True)
            error_path = cut_video_moviepy(st.session_state.visulization_path, start_frame, end_frame, 'UserInterface/outputs/visualizations/error/error.mp4')
            col1, _ = st.columns(2)
            with col1:
                st.video(error_path)
        else:
            st.markdown('## Squat errors')
            st.markdown('**No errors detected**: :white_check_mark:')

        x_position_to_plot = st.multiselect('Select the horizontal position to plot', front_evaluator.joints_id_dict.keys(), default=['left_knee', 'right_knee', 'left_shoulder', 'right_shoulder'])
        if x_position_to_plot:
            plot_x_position(front_evaluator, x_position_to_plot, squat_number)
        y_position_to_plot = st.multiselect('Select the vertical position to plot', front_evaluator.joints_id_dict.keys(), default=['left_shoulder', 'right_shoulder'])
        if y_position_to_plot:
            plot_y_position(front_evaluator, y_position_to_plot, squat_number)
        

def main(json_file_path, person_id):

    point_of_view = st.selectbox('Choose a point of view', ['Left side', 'Right side', 'Front'])
    st.session_state['point_of_view'] = point_of_view

    if point_of_view == 'Left side' or point_of_view == 'Right side':
        side_squat_analyzer(json_file_path, person_id, point_of_view)

    elif point_of_view == 'Front':
        front_squat_analyzer(json_file_path, person_id)