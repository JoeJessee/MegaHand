function [PredictedAction] = classifyEMG(x) %#codegen
% This function extracts predetermined features from raw EMG signal of 8
% channels and uses a pretrained classifier to classifer the hand position
% corresponding to the EMG signals.

% Retrieve the compact classifier model
EMGClassifier = loadCompactModel('myClassifier.mat');

% Extract the features from the signal
feat = generateFeatures(x);

% Apply the EMG classifier and retrieve the correct label
n = predict(EMGClassifier,feat);
labels = {...
    'Chuck Grip','Fine Pinch','H. Open','Hook Grip','Key Grip',...
    'No Move','Power Grip','Thumb Enclosed','Tool Grip','W. Abduction',...
    'W. Adduction','W. Extension','W. Flexion','W. Pronation','W. Supination'};
PredictedAction = labels{n};
end
