import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Read Data
data = pd.read_csv(r'C:\Users\pattersonrb\PyProjects\MegaHand\EMG_Classification_Matlab\Data\TrainingData\TrainingDataset.txt')
y_train = data.Action.values.reshape(-1, 1)
X_train = data.drop('Action', axis=1).values

# Test Data

# Generate Model
model = GaussianNB()
model.fit(X, y)
