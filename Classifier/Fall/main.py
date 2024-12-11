import os
import sys
sys.path.append(os.getcwd())
import pickle
import numpy as np
from Classifier.Fall.data_preparation import WindowFeatureExtractor
from Classifier.Fall.train import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 

def main():


    poses_sequence_path = 'Data/Fall/Dataset_CAUCAFall/Poses_sequences/'
    file_list = os.listdir(poses_sequence_path)
    file_list = [poses_sequence_path + file_name for file_name in file_list]

    ### Model and grid search parameters ###
    class_weight = {'Normal': 1, 'Fall': 5, 'Lying down': 3}
    model = RandomForestClassifier(class_weight=class_weight)

    window_param_grid = {
        'window_size': [5, 10, 20, 30, 40, 50, 70, 80],
        'step_size': [2, 5, 10, 20, 30, 40]
    }

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    trainer = ModelTrainer(model, param_grid, window_param_grid)

    # first we need to find the best window parameters
    #best_window_params = trainer.window_grid_search(window_param_grid, file_list)
    window_size = 10
    step_size = 5
    window_feature_extractor = WindowFeatureExtractor(window_size=window_size, step_size=step_size)

    # then we can create our training and test set, and make a grid search over the model parameters
    X, y = window_feature_extractor.prepare_data(file_list)
    nan_index = np.where(y=='nan')[0]
    X = X.drop(nan_index, axis=0)
    y = np.delete(y, nan_index, axis=0)
    print(X.shape, y.shape)
    print(np.unique(y, return_counts=True))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # drop samples with NaN values


    # we can now perform a grid search over the model parameters
    #best_param = trainer.grid_search(param_grid, X_train_scaled, y_train)
    best_param = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}

    # we can now train the model with the best parameters
    model = RandomForestClassifier(class_weight=class_weight, **best_param)
    model.fit(X_train_scaled, y_train)
    print(model.classes_)

    # and evaluate it on the test set
    print("Test score:", model.score(X_test_scaled, y_test))

    # Finally we can save the model, and the scaler
    
    pickle.dump(model, open('Classifier/Fall/models/randomForest_window_{}_step_{}.sav'.format(window_size, step_size), 'wb'))
    pickle.dump(scaler, open('Classifier/Fall/models/standardScaler_window_{}_step_{}.sav'.format(window_size, step_size), 'wb'))


if __name__ == '__main__':
    main()