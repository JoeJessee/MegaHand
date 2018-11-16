""" A Script for training a machine learning model on data
"""

import pandas as pd
import numpy as np
from EDA import glob_data
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

def concat_files(iterable):
    """Concatenates all files in iterable into a single data frame
    Resets index along data frame
    Assumes column names are the same in all files

    Arguments:
    ----------
    iterable - Any iterable
    List of file paths to data files

    Returns:
    --------
    df: pandas.core.frame.DataFrame
    Dataframe containing all files in a single frame
    """
    try:
        iterator = iter(iterable)
    except TypeError:
        print('Concat_files requires filepaths to be in an iterable')
    data = []
    for file in iterable: 
        data.append(pd.read_csv(file))
    df = pd.concat(data, ignore_index=True)
    return df

if __name__ == '__main__':
    # Read Data
    data_train = pd.read_csv(r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TrainingData\TrainingDataset.txt')
    y_train = data_train.Action.values
    X_train = data_train.drop('Action', axis=1).values

    # Test Data
    file_list = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TestingData')
    data_test = concat_files(file_list)
    y_test = data_test.Action.values
    X_test = data_test.drop('Action', axis=1).values

    # Establish pipeline
    pl = Pipeline([('int', PolynomialFeatures(include_bias=False, interaction_only=True)),
                   ('scale', MinMaxScaler()),
                   ('select', SelectKBest(chi2)), 
                   ('clf', GradientBoostingClassifier())
                   ])
    
    # establish gridsearchcv, cv=3 to save on computation
    param_grid = {'select__k': [5, 7, 10]}
    cv = GridSearchCV(pl, param_grid=param_grid, cv=3)

    # train and retrieve best_parameters
    cv.fit(X_train, y_train)
    print(cv.best_params_)
    model = cv.best_estimator_

    # predict and score
    y_predict = model.predict(X_test)
    print(model.score(X_test, y_test))
    report = pd.DataFrame.from_dict(classification_report(y_test, y_predict, output_dict=True), orient='index')
    report.to_csv(r'c:\users\pattersonrb\pyprojects\megahand\models\int_MinMax_kbest_GBC.csv')

    # pickle model
    with open(r'c:\users\pattersonrb\pyprojects\megahand\models\int_MinMax_kbest_GBC.pickle', 'wb') as file:
        pickle.dump(model, file)