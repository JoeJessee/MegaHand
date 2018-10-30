"""
A module for globbing all files of a specific type in a specific directory
"""

from glob import iglob
from os import getcwd
from more_itertools.more import peekable

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

if __name__ == '__main__':
    files = glob_data(folder='C:\\Users\\pattersonrb\\PyProjects\\MegaHand\\EMG_Classification_Matlab\\Data\\TrainingData')