"""
A module for exploratory data analysis. 

This module contains a number of functions to aid with EDA, including:
glob_data: collect all data files into an iterator
pair_grid: graph pairwise relations with columns of numeric data
plot_distribution: plot distribution of data against standard curves

Example use:
files = glob_data(extension='.xlsx', folder='C:/Users/your_name/your_data')
"""

#%%
from glob import iglob
from os import getcwd
from more_itertools.more import peekable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        if ('/' not in folder) and ('\\' not in folder):
            raise ValueError('Path should be specified with / or \\ only')
        if folder.endswith(('/', '\\')):
            raise ValueError('Path should not end with a separator')
    files = peekable(iglob(f'{folder}\\*{extension}'))
    if files.peek('empty') == 'empty': # Returns empty if files contains no items
        raise ValueError(f'No {extension} files found at {folder}')
    return files

#%%
def pair_grid(iterator, diag_type=plt.hist, off_type=plt.plot):
    """Plots all data files in interator as seaborn pair grids and saves the images
    Currently only supports .csv files, and directly specified graph types

    Arguments:
    ----------
    iterator: iterable, required
        Contains a list of path names to data files
        Any iterable may be used (list, tuple, generator, etc.). 
    diag_type: function, default=plt.hist
        In: [plt.hist, sns.kdeplot]
        Type of plot to be used on diagonal
    off_type: function, default=plt.plot
        In: [plt.plot, plt.hexbin, sns.kdeplot]
        Type of plot to be used for all plots on diagonal

    Returns:
    --------
    figures: generator
        Generator containing plotted figures

    Example:
    --------
    figures = pair_grid(['filea', 'fileb', filec'], diag_type='kde', off_type='hex')
    """
    sns.set(style='ticks', font='monospace')
    diag_type_allowed = [plt.hist, sns.kdeplot]
    off_type_allowed = [plt.plot, plt.hexbin, sns.kdeplot]
    if (not callable(diag_type)) or (not callable(off_type)):
        raise TypeError('diag_type and off_type must be functions')
    if diag_type not in diag_type_allowed:
        raise ValueError(f'diag_type needs to be in {diag_type_allowed}')
    if off_type not in off_type_allowed:
        raise ValueError(f'off_type needs to be in {off_type_allowed}')
    for file in iterator:
        data = pd.read_csv(file)
        g = sns.PairGrid(data)
        g = g.map_diag(diag_type)
        if off_type == plt.plot:
            g = g.map_offdiag(off_type, marker='.', linestyle='none')
        else:
            g = g.map_offdiag(off_type)
    yield g