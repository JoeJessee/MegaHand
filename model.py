import pandas as pd
import numpy as np
from EDA import glob_data
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    pl = Pipeline([('scale', RobustScaler()),
                   ('pca', IncrementalPCA(n_components=7, whiten=True)),
                   ('clf', GradientBoostingClassifier())
                   ])
    
    # establish gridsearchcv, cv=3 to save on computation
    # param_grid = {'clf__n_estimators': [int(i) for i in np.linspace(100, 1100, 5)]}
    # cv = gridsearchcv(pl, param_grid=param_grid, cv=3)

    # train and retrieve best_parameters
    pl.fit(X_train, y_train)
    # print(cv.best_params_)
    # model = cv.best_estimator_

    # predict and score
    y_predict = pl.predict(X_test)
    print(pl.score(X_test, y_test))
    report = pd.DataFrame.from_dict(classification_report(y_test, y_predict, output_dict=True), orient='index')
    report.to_csv(r'c:\users\pattersonrb\pyprojects\megahand\models\robustscaler_ipca_gbc.csv')

    # pickle model
    with open(r'c:\users\pattersonrb\pyprojects\megahand\models\robustscaler_ipca_gbc.pickle', 'wb') as file:
        pickle.dump(pl, file)
    