## Breast Cancer Classification using Neural Network

This project demonstrates how to use a neural network to classify breast cancer tumors as either malignant or benign based on various medical features. We use the **Breast Cancer dataset** from sklearn, preprocess the data, standardize it, and then train a deep learning model using **TensorFlow/Keras**.

### Project Overview

The dataset contains features derived from a digitized image of a breast mass, including various measurements of the cells present in the image. We use a neural network to predict the malignancy (0 for malignant, 1 for benign) of the tumor. After training the model, we evaluate its performance and make predictions on new input data.

### Steps Involved:

1. **Data Exploration and Preprocessing**:
   - The dataset is loaded and inspected. We check for missing values, the shape of the data, and the distribution of the target variable (`label`), which indicates if a tumor is malignant or benign.
   - We standardize the feature data using **StandardScaler** to normalize it before training the neural network.

2. **Splitting the Data**:
   - The dataset is split into training and testing sets using **train_test_split** from sklearn. This ensures that the model can be trained on one subset of the data and tested on another to evaluate its performance.

3. **Building the Neural Network**:
   - A simple **Sequential model** in **Keras** is used to create the neural network.
   - The model consists of a **Flatten** input layer, a **Dense** hidden layer with ReLU activation, and a **Dense** output layer with a **Sigmoid** activation function to predict the binary outcome (malignant or benign).

4. **Model Compilation and Training**:
   - The model is compiled with the **Adam optimizer** and the **sparse categorical cross-entropy loss function** for binary classification.
   - The model is trained for 10 epochs, with the accuracy and loss being plotted over the training and validation data for each epoch.

5. **Model Evaluation and Prediction**:
   - After training, the model is evaluated on the test data to determine its accuracy.
   - Predictions are made on test data, and the predicted labels are compared to the true labels.
   - A custom input is provided to the model to predict the malignancy of a new tumor based on given features.

