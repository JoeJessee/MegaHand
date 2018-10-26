# MegaHand
Robotic Hand Project Repository

## Pseudocode outline

1. Model training/selection
    - All kinds of options here! See the `scikit-learn` [documentation](http://scikit-learn.org/stable/model_selection.html#model-selection$) for a discussion of models and model selection.
2. Collect raw data
    - This would be where we need some sort of EMG reader
3. Pre-process Data
    - See [here](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing) for all available options. Probably won't need everything (*eg* don't think we will need an encoder as all our data is numerical).
4. Lineaer classifier
5. Convert model output to actuator code
    - In my head, I'm visualising that the linear classifier will give us a categorical output, say 'key grip', and we will have a class that takes that category and converts it to whatever the actuator needs.
    - Can we even talk to the actuators in python?