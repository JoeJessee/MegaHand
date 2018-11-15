import pandas as pd
import numpy as np
from EDA import glob_data
from sklearn.naive_bayes import GaussianNB

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
    df = pd.DataFrame()
    for file in iterable: 
        df.append(pd.read_csv(file), ignore_index=True)
    return df

if __name__ == '__main__':
    # Read Data
    data = pd.read_csv(r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TrainingData\TrainingDataset.txt')
    y_train = data.Action.values.reshape(-1, 1)
    X_train = data.drop('Action', axis=1).values

    # Test Data
    file_list = glob_data(r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TestingData')
    data_test = concat_files(file_list)

    print(data_test)