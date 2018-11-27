""" A Script for training a machine learning model on data
Standard pipelines and GridSearchCV are used.

The pipeline elemets were selected by scoring multiple elements with minimal tuning. 
The RobustScaler is less effected by outliers than other options.
PolynomialFeatures adds interaction terms
GradientBoostingClassifiers perform better (generally) than the equivalent RandomForest

Within GBC, parameters were chosen as follows:
High n_estimators with early stopping finds a good balance between computation time and performance
    by preventing overfitting
Presorting increases computation speed
Subsampling, leading to stochastic GBC, increases speed while helping to prevent overfitting
    Value of 0.5 is standard
Decreasing max_features decreases variance and time, but increases bias.
    'sqrt' is middle ground between 'log2' and 'none'
Learning rate (shrinkage) < 1, and prefereably < 0.1, drastically increases performance at cost of time
max_depth limits the number of nodes in the trees. The range 4 <= x <= 8 is considered ideal.

Functions:
----------
concat_files(iterable) - reads in all files in iterable anc concatenates them into a single dataframe
"""

import pandas as pd
import numpy as np
from EDA import glob_data
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def concat_files(iterable):
    """Concatenates all files in iterable into a single data frame
    Resets index along data frame
    Assumes column names are the same in all files

    Arguments:
    ----------
    iterable: Any iterable(list, generator, tuple, etc)
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

def save_csv(data, path="data.csv"):
    """ Saves input data as a csv to path
    
    Arguments:
    ----------
    data: pd.DataFrame, np.ndarray, dict, list
        Data to be written to csv file.
        For dict, keys are column names and values are column values
        For list, each element is treated as a row
    path: str, default: 'data.csv'
        String containing path to desired save file. 
        Must end in '.csv.'

    Returns:
    --------
    True, if successful

    """
    if type(data) == pd.core.frame.DataFrame

if __name__ == '__main__':
    # Read Data
    # Folder path should be location of training data on your system
    data_train = pd.read_csv(r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TrainingData\TrainingDataset.txt')
    y_train = data_train.Action.values
    X_train = data_train.drop('Action', axis=1).values

    # Test Data
    # Folder path should be location of testing data on your system
    file_list = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TestingData')
    data_test = concat_files(file_list)
    y_test = data_test.Action.values
    X_test = data_test.drop('Action', axis=1).values

    # Establish pipeline
    pl = Pipeline([('int', PolynomialFeatures(include_bias=False, interaction_only=True)),
                   ('scale', RobustScaler()),
                   ('clf', GradientBoostingClassifier(
                        n_estimators=1000, n_iter_no_change=5, 
                        tol=0.001, validation_fraction=0.2, presort=True, 
                        subsample=0.5, max_features='sqrt')
                    )])
    
    # establish gridsearchcv, cv=3 to save on computation
    param_grid = {'clf__learning_rate': [0.001, 0.01, 0.1, 0.5],
                  'clf__max_depth': [4, 6, 8]}
    cv = GridSearchCV(pl, param_grid=param_grid, cv=3)

    # train and retrieve best_parameters
    pl.fit(X_train, y_train)
    #print(cv.best_params_)
    model = pl

    # predict and score
    y_predict = model.predict(X_test)
    print(model.score(X_test, y_test))
    report = pd.DataFrame.from_dict(classification_report(y_test, y_predict, output_dict=True), orient='index')
    report.to_csv(r'c:\users\pattersonrb\pyprojects\megahand\models\final_1.csv')

    # pickle model
    with open(r'c:\users\pattersonrb\pyprojects\megahand\models\final_1.pickle', 'wb') as file:
        pickle.dump(model, file)