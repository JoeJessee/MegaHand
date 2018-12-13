
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import os
import glob
path = 'c:\\'
extension = 'csv'
os.chdir(path= "/Users/noahlevi/MegaHand-1/Models")
models = [i for i in glob.glob('*.{}'.format(extension))]
print(models)


#%%
def model_stats(models):
    precision=[pd.read_csv(i).iloc[-1,1] for i in models]
    recall=[pd.read_csv(i).iloc[-1,2] for i in models]
    f1=[pd.read_csv(i).iloc[-1,3] for i in models]
    labels=[i[:-4] for i in models]
    
    plt.bar(labels, precision, color="red")
    plt.xticks(rotation=65)
    plt.xlabel("Model")
    plt.ylabel("True Positives/ Total Positives")
    plt.title("Precision")
    plt.show()
    plt.bar(labels, recall, color= "blue")
    plt.xticks(rotation=65)
    plt.xlabel("Model")
    plt.ylabel("True Positives/ False Negatives")
    plt.title("Recall")
    plt.show()
    plt.bar(labels, f1, color="green")
    plt.xticks(rotation=65)
    plt.xlabel("Model")
    plt.ylabel("F1_Score")
    plt.title("F1_Score")
    plt.show()
    
        


#%%
model_stats(models)


