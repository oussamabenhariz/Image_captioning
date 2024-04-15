# Image_captioning
This project aims to generate descriptive captions for images using a combination of a pre-trained VGG16 model for feature extraction and a Long Short-Term Memory (LSTM) neural network for caption generation.

## Overview
The project utilizes the Flickr8k dataset for training and testing.
It employs transfer learning with VGG16 to extract features from images.
LSTM-based architecture is used to generate captions based on the extracted features.
## Project Structure
image_cap.ipynb: Jupyter Notebook containing the code.
kaggle.json: Kaggle API key for dataset download.
data/: Directory containing dataset and saved features.
model.h5: Trained model saved in HDF5 format.
## Components
### 1. Data Preparation

Download and preprocess the Flickr8k dataset.
Extract features using the VGG16 model and save them.
### 2. Text Preprocessing

Clean and tokenize captions.
Prepare data for training.

### 3. Model Building

Design an architecture combining VGG16 features with LSTM layers.
Compile the model with appropriate loss and optimizer.
### 4. Training

Train the model using the prepared data.
Monitor training progress and adjust hyperparameters if necessary.
### 5. Evaluation

Evaluate the model's performance using BLEU scores on test data.
### 6. Prediction

Generate captions for new images using the trained model.
Visualize actual and predicted captions for sample images.
## Results
The trained model achieves a BLEU-1 score of X and a BLEU-2 score of Y on the test dataset.
## Usage
Clone the repository and navigate to the project directory.
Ensure the required dependencies are installed.
Run the Jupyter Notebook image_cap.ipynb to train the model and generate captions.
## Sample Predictions
Include sample images with their actual and predicted captions.
