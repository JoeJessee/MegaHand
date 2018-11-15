import pandas as pd
import numpy as np
from EDA import glob_data
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
    y_train = data_train.Action.values.reshape(-1, 1)
    X_train = data_train.drop('Action', axis=1).values

    # Test Data
    file_list = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TestingData')
    data_test = concat_files(file_list)
    y_test = data_test.Action.values.reshape(-1, 1)
    X_test = data_test.drop('Action', axis=1).values