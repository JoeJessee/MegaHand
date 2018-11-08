"""
A module for exploratory data analysis. 

This module contains a number of functions to aid with EDA, including:
glob_data: collect all data files into an iterator
pair_grid: graph pairwise relations with columns of numeric data
plot_distribution: plot distribution of data against standard curves

Example use:
files = glob_data(extension='.xlsx', folder='C:/Users/your_name/your_data')
figs = pair_grid()
"""

#%%
from glob import iglob
from os import getcwd
from more_itertools.more import peekable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
def glob_data(extension='.csv', folder=getcwd()):
    """Globs data files in a folder into an iterator

    Arguments:
    ----------
    extension: str, default='.csv'
        File type of data files. Currently does not support mixed types
        Should contain leading .
    folder: str, default=current working directory
        Path name to data. User specified paths should be delimited with
        / or \\ and should not end with a separator
    
    Returns:
    --------
    files: generator containing all data files found in the folder
    """
    if type(extension) != str:
        raise TypeError('Extension must be a string')
    if type(folder) != str:
        raise TypeError('Folder must be a string')
    if folder != getcwd():
        if ('\\' not in folder):
            raise ValueError('Path should be specified with / or \\ only')
        if folder.endswith(('\\')):
            raise ValueError('Path should not end with a separator')
    files = peekable(iglob(fr'{folder}\*{extension}'))
    if files.peek('empty') == 'empty': # Returns empty if files contains no items
        raise ValueError(f'No {extension} files found at {folder}')
    return files

#%%
def corr_matrix(iterator):
    """Generate a diagonal correlational matrix for a dataframe
    Altered from: seaborn.pydata.org/examples/many_pairwise_correlations.html
    
    Arguments
    ---------
    iterator: iterable containing data file names

    Yields
    ------
    Iterable containing:
    ax: matplotlib ax element of correlation matrix

    Example
    -------
    figs = corr_matrix((filea, fileb, filec))
    """
    sns.set(style='white', font='monospace')
    for data in iterator:
        df = pd.read_csv(data)
        corr = df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        ax = sns.heatmap(corr, mask=mask, 
                         cmap='seismic', vmin=-1, vmax=1,
                         cbar_kws={'shrink': 0.5}, 
                         square=True, linewidths=0.5)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(data.split('\\')[-1], fontdict={'fontsize': 12, })
        yield ax

def ECDF(iterable):
    """Plots ECDF of all numeric columns in a dataframe
    """


#%%
files = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TrainingData')
plots = corr_matrix(files)
for i in plots:
    plt.show()