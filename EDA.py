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
from matplotlib.backends.backend_pdf import PdfPages

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
    
    Example:
    --------
    files = globdata(extension='.txt', folder='/my/favorite/dir)
    """
    if type(extension) != str:
        raise TypeError('Extension must be a string')
    if not extension.startswith('.'):
        raise ValueError('Extension must start with "."')
    if type(folder) != str:
        raise TypeError('Folder must be a string')
    """if folder != getcwd():
        if ('\\' not in folder):
            raise ValueError('Path should be specified with / or \\ only')
        if folder.endswith(('\\', '/')):
            raise ValueError('Path should not end with a separator')"""
    files = peekable(iglob(fr'{folder}\*{extension}'))
    if files.peek('empty') == 'empty': # Returns empty if files contains no items
        raise ValueError(f'No {extension} files found at {folder}')
    return files

#%%
def corr_matrix(iterable):
    """Generate a diagonal correlational matrix for a dataframe
    Altered from: seaborn.pydata.org/examples/many_pairwise_correlations.html
    
    Arguments
    ---------
    iterator: iterable containing data file names
    List, tuple, generator, etc.

    Yields
    ------
    Generator containing:
    fig: matplotlib fig element of correlation matrix

    Example
    -------
    figs = corr_matrix(('filea', 'fileb', 'filec'))
    """
    try:
        iterator = iter(iterable)
    except TypeError:
        print('Input to corr_matrix must be an iterable collection of filepaths.')
    sns.set(style='white', font='monospace')
    for data in iterable:
        df = pd.read_csv(data).select_dtypes(include='number')
        corr = df.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots()
        sns.heatmap(corr, mask=mask, ax=ax,
                    cmap='seismic', vmin=-1, vmax=1,
                    cbar_kws={'shrink': 0.5}, 
                    square=True, linewidths=0.5)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(data.split('\\')[-1], fontdict={'fontsize': 12, })
        yield fig

def ECDF(iterable):
    """Plots ECDF of all numeric columns for files in iterable

    Arguments
    ---------
    iterable containing data file names

    Yields
    ------
    Generator containing:
    fig: matplotlib fig element of ECDF, 1 per file

    Example
    -------
    ecdfs = ECDF(('filea', 'fileb', 'filec'))
    """
    try:
        iterator = iter(iterable)
    except TypeError:
        print('Input to ECDF must be an iterable collection of filepaths.')
    sns.set(style='ticks', font='monospace')
    for file in iterable:
        data = pd.read_csv(file).select_dtypes(include='number')
        fig, ax = plt.subplots()
        for col in data.columns:
            ax.plot(np.sort(data[col]), np.arange(1, len(data[col]) + 1) / float(len(data[col])), label=col)
            sns.despine(offset=5)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title(file.split('\\')[-1], fontdict={'fontsize': 12})
        ax.legend(fontsize=8, frameon=False)
        yield fig

def figs_to_pdf(iterable, filename='___.pdf'):
    """Saves an iterable containing matplotlib figs as a PDF

    Arguments
    ---------
    iterable: iterable containing matplotlib plot/figs
    filename: type = str, default = '___.pdf'
        name of pdf to be created

    Returns
    -------
    True, if successful

    Example
    -------
    figs_to_pdf((fig1, fig2, fig3), 'test.pdf')
    """
    try:
        iterator = iter(iterable)
    except TypeError:
        print('Input to figs_to_pdf must be an iterable collection of matplotlib figs.')
    if type(filename) != str:
        raise TypeError('filename must be a string')
    if '.pdf' not in filename:
        raise ValueError('filename must be a .pdf file')
    with PdfPages(filename) as pdf:
        for img in iterable:
            pdf.savefig(img)
    return True

#%%
if __name__ == '__main__':
    files = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\TrainingData')
    matrices = corr_matrix(files)
    files = glob_data(folder=r'C:\Users\pattersonrb\PyProjects\MegaHand\TrainingData')
    plots = ECDF(files)
    _ = figs_to_pdf(matrices, r'C:\Users\pattersonrb\PyProjects\MegaHand\TrainingData\corr_matrices.pdf')
    _ = figs_to_pdf(plots, r'C:\Users\pattersonrb\PyProjects\MegaHand\TrainingData\ECDFs.pdf')