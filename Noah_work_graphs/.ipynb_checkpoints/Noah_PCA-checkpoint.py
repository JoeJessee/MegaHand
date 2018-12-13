
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA 
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

import os
import glob
path = 'c:\\'
extension = 'csv'
os.chdir(path= "/Users/noahlevi/MegaHand-1/EMG_Classification_Matlab/Data/TrainingData/")
Training_Data_Files = [i for i in glob.glob('*.{}'.format(extension))]

print(Training_Data_Files)

'''Variance of PCA'''
def PCA_Variance(x):
    data=pd.read_csv(x).iloc[0:,0:8]
    scaler = RobustScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler,pca)
    pipeline.fit(data)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('Variance')
    plt.title(x[:-4])
    plt.show()
    
for i in Training_Data_Files:
    PCA_Variance(i)
    


