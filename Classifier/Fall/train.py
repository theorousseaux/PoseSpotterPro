import os
import sys
sys.path.append(os.getcwd())
from Classifier.Fall.data_preparation import WindowFeatureExtractor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 

class ModelTrainer():

    def __init__(self, model, param_grid, window_param_grid, cv_folds=5):
        self.model = model

        self.cv_folds = cv_folds

        self.window_param_grid = window_param_grid
        self.param_grid = param_grid
        
        self.best_window_params = None
        self.best_params = None
        

    
    def window_grid_search(self, window_param_grid, list_of_files):
        """
        Performs a grid search over the window parameters to find the best window parameters.
        """

        print("Performing window grid search...")

        best_window_params = None
        best_score = 0

        for window_size in window_param_grid['window_size']:
            for step_size in window_param_grid['step_size']:
                print("Window size:", window_size)
                print("Step size:", step_size)

                window_feature_extractor = WindowFeatureExtractor(window_size, step_size)

                X, y = window_feature_extractor.prepare_data(list_of_files)
                X_scaled = StandardScaler().fit_transform(X)

                scores = cross_val_score(self.model, X_scaled, y, cv=self.cv_folds, n_jobs=-1)
                print("Scores:", scores)

                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_window_params = {'window_size': window_size, 'step_size': step_size}

        self.best_window_params = best_window_params
        print('\n--------------------------')
        print("Best window parameters:", self.best_window_params)
        print("Best window score:", best_score)

        return best_window_params
    

    def grid_search(self, param_grid, X_train, y_train):
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=self.cv_folds, verbose=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        print("Best parameters:", self.best_params)
        print("Best score:", grid_search.best_score_)
        return grid_search.best_params_



if __name__ == '__main__':
    pass