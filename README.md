# Deep Learning Throat Diagnosis

A throat image classification project using neural networks (Custom CNN and VGG16). Built with TensorFlow, Keras, scikit-learn, Pandas, and PIL. The repository includes two interactive dashboards: one for analytics (training results) and another for predictions (inflammation diagnosis based on uploaded images).

## Project Description
This project aims to classify throat images into two categories
- **Pharyngitis**
- **No Pharyngitis**

Two modeling approaches are implemented
1. **Custom CNN** – A simple, self-built convolutional neural network.
2. **Transfer Learning with VGG16** – A pre-trained VGG16 model with additional custom layers.

## Features
- **Training Results Dashboard** Visualizations of accuracy, loss, and detailed classification reports.
- **Prediction Dashboard** Allows users to upload throat images, which are classified as “Pharyngitis” or “No Pharyngitis” with associated probabilities.

## Project Steps
### Data Preparation:
- Splitting the dataset into training, validation, and test sets.
- Scaling images to the [0,1] range and resizing them to 100x100 pixels.
- Data augmentation (rotation, shifting, scaling) to increase dataset diversity.

### Model Development
1. **Custom CNN**
   - Architecture: Convolutional layers, pooling, dropout, and a dense output layer with sigmoid activation.
   - Optimizer: Adam, Loss Function: `binary_crossentropy`.

2. **Transfer Learning with VGG16**
   - Frozen weights from the pre-trained VGG16 model.
   - Added layers: GlobalAveragePooling2D, BatchNormalization, Dropout.
   - Optimizer: Adam with dynamic learning rate (`ReduceLROnPlateau`), Loss Function: `binary_crossentropy`.

### Validation and Testing:
- Model evaluation on validation and test sets.
- EarlyStopping to prevent overfitting.
- Metrics: Accuracy, Loss, Classification Reports.

### Visualization and Analysis:
- Interactive plots of training results.
- Performance comparison between the two approaches.

## Technical Requirements
- **Python 3.7+**
- Libraries: TensorFlow, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, PIL, Dash.

## Summary
This project demonstrates the practical application of deep learning in medical image analysis, providing tools to visualize training results and classify new throat images. It combines the simplicity of a custom CNN model with the power of transfer learning to deliver an optimized, scalable solution.

## Potential Applications
- Supporting medical diagnostics in clinical settings.
- Developing mobile apps for early throat disease detection.
- Research and education in deep learning for medical imaging.

# Data: 
To download the data needed for the project, click on the link below

https://drive.google.com/drive/folders/1KxByAadRtLKBV2TF3wpJXcqtmisGFpnr?usp=sharing

# Interactive dashboards:

-  Dashboard application (App 1)](https://deep-learning-throat-diagnosis-app1.streamlit.app)
-  Dashboard training (App 2)(https://deep-learning-throat-diagnosis-app2.streamlit.app)

# A video explaining how the code works in applications:

https://drive.google.com/drive/folders/16doibOU-Yr54VWPM5iYsVmfeXgDSSgcN?usp=sharing