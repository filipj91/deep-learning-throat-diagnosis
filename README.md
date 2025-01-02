# deep-learning-throat-diagnosis
Throat image classification project using neural networks (Custom CNN and VGG16). TensorFlow, Keras, scikit-learn, Pandas and PIL were used. The repository includes 2 dashboards: analytical (training results) and predictive (inflammation diagnosis based on the image). An example of ML in medicine
Project Description:
This project aims to develop a system for classifying throat images using deep neural networks. The system allows classification of images into two categories:

Pharyngitis (pharyngitis)
No pharyngitis (pharyngitis)
The project uses two modeling approaches:

Custom CNN: A simple, self-built convolutional model (CNN).
Transfer Learning with VGG16: Using a pre-trained VGG16 model with additional layers tailored to the specific classification problem.
Functionalities
The project offers two interactive dashboards:


Dashboard of training results:
Visualizations of accuracy, loss, and detailed classification reports for each model.
Prediction Dashboard:
Allows the user to upload a throat image, which is automatically classified as “pharyngitis” or “no pharyngitis.” The prediction result is presented along with the probability.
Project steps
Data preparation:

Dividing the collection into training, validation and test data.
Scaling images to the range [0,1] and resizing to 100x100 pixels.
Data augmentation (rotation, shifting, scaling) to increase the diversity of the collection.
Model construction:

Custom CNN:
Architecture: convolutional layers, pooling, dropout and dense output layer with sigmoid activation.
Optimizer: Adam
Loss function: binary_crossentropy

Transfer Learning with VGG16:
Using the frozen weights of the pre-trained VGG16 model.
Adding GlobalAveragePooling2D, BatchNormalization and Dropout layers.
Optimizer: Adam with dynamic learning rate (ReduceLROnPlateau).
Loss function: binary_crossentropy.
Validation and testing:


Evaluating models on a validation and test set.
Using the EarlyStopping technique to avoid model overfitting.
Determination of performance metrics such as:
Accuracy (accuracy)
Loss (loss)
Classification report and detailed results for each class.
Visualization and analysis:

Interactive graphs of training results.
Comparison of performance of both modeling approaches.
Exporting results:

Training results are saved in CSV format (training_metrics.csv) for further analysis.
Technical requirements
Python version: 3.7+
Python libraries:
TensorFlow
NumPy
Pandas
scikit-learn
Matplotlib
Seaborn
PIL (Pillow)
Dash
Summary
This project demonstrates the practical application of deep learning in medical image analysis, providing users with tools to visualize training results and classify new throat images. Combining the simplicity of the in-house CNN model with the power of transfer learning enables the analysis of the advantages of both approaches, offering a solution optimized for accuracy and scalability.

Potential applications
The system can find applications in:

Supporting medical diagnostics in clinical settings.
Mobile application development for early detection of throat diseases.
Medical education and deep learning model research.

Data:
To download the data needed for the project, click on the link below:

https://drive.google.com/drive/folders/1KxByAadRtLKBV2TF3wpJXcqtmisGFpnr?usp=sharing

Interactive dashboards:

-  Dashboard application (App 1)](https://deep-learning-throat-diagnosis-app1.streamlit.app)
-  Dashboard training (App 2)(https://deep-learning-throat-diagnosis-app2.streamlit.app)

A video explaining how the code works in applications:

1. [app1/video.mp4](app1/video.mp4) - video explaining how the app works 1.
2. [app2/video.mp4](app2/video.mp4) - Video explaining how the app works 2.

App 1 
<video src="app1/video.mp4" controls width="500"></video>
App 2 
<video src="app2/video.mp4" controls width="500"></video>