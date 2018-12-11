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
import matplotlib.pyplot as plt
import itertools
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
def plot_confusion_matrix(cm, classes,
          normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Modified from : scikit-learn.org example code at: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    # Read Data
    # Folder path should be location of training data on your system (Add Directory)
    data_train = pd.read_csv(r'')
    y_train = data_train.Action.values
    X_train = data_train.drop('Action', axis=1).values

    # Test Data
    # Folder path should be location of testing data on your system (Add Directory)
    file_list = glob_data(folder=r'')
    data_test = concat_files(file_list)
    y_test = data_test.Action.values
    X_test = data_test.drop('Action', axis=1).values
    labels = data_test.columns

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
    cv.fit(X_train, y_train)
    print(cv.best_params_)
    model = cv.best_estimator_

    # predict and score (Add directory)
    y_predict = model.predict(X_test)
    print(model.score(X_test, y_test))
    report = pd.DataFrame.from_dict(classification_report(y_test, y_predict, output_dict=True), orient='index')
    report.to_csv(r'')
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    plt.figure()
    plot_confusion_matrix(cm, classes=labels,
    title='Confusion matrix, without normalization')

    plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=labels, normalize=True,
    title='Normalized confusion matrix')

    plt.show()

    # pickle model (Add Directory)
    with open(r'', 'wb') as file:
        pickle.dump(model, file)
 


