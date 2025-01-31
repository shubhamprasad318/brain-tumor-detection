# **Brain Tumor Detection System**

## **Overview**

This project is a web-based brain tumor detection system using deep learning. It utilizes convolutional neural networks (CNNs) for binary and multi-class classification of brain MRI images. The system allows users to upload MRI scans and predicts whether the image contains a brain tumor or not.

## **Features**

* **Image Upload**: Users can upload MRI images via a web interface.  
* **Tumor Detection**: The system uses two CNN models for prediction.  
* **Ensemble Technique**: The predictions from both models are combined for better accuracy.  
* **Web-based Interface**: The frontend is built using HTML, CSS, and JavaScript.  
* **Flask Backend**: The backend is developed in Python using Flask.  
* **Visualization**: Displays the uploaded image and prediction results.

## **Dataset**

The dataset consists of MRI images categorized into two classes:

1. **Tumor** \- Images containing brain tumors.  
2. **Without Tumor** \- Images with no tumors.

The dataset is structured as follows:

/dataset  
   /train  
      /tumor  
      /without\_tumor  
   /valid  
      /tumor  
      /without\_tumor

dataset link- https://drive.google.com/drive/folders/1yfg_eOKelB82PiIqtfAdh0qHFbQ92QVo?usp=drive_link
## **Models**

Two CNN models are used for classification:

1. **Model 1**: `best_model.keras`  
   * Input size: `(224, 224, 3)`  
   * Architecture: Custom CNN with multiple convolutional layers.  
   * Activation: Sigmoid for binary classification.  
2. **Model 2**: `best_modelwithtransferlearning.keras`  
   * Input size: `(400, 400, 3)`  
   * Architecture: Transfer learning-based CNN.  
   * Activation: Sigmoid for binary classification.

model1 link:- https://drive.google.com/file/d/1lCKHvx0aJirlufl2AWAqfqbNAfXoLT6S/view?usp=drive_link
model2 link:- https://drive.google.com/file/d/1--Jt7qX4YUilLteS2anHu2vQhxjUGpRm/view?usp=drive_link
### **Preprocessing**

* Resize images to the required input size.  
* Normalize pixel values to `[0,1]`.  
* Expand dimensions for model input.

## **Web Application**

### **Frontend**

The frontend allows users to:

* Upload MRI images.  
* View the image preview.  
* Click a **Predict** button to analyze the image.  
* See the prediction results.

### **Backend (Flask API)**

The backend:

* Accepts image uploads.  
* Processes and normalizes the image.  
* Feeds the image into both CNN models.  
* Combines predictions using an ensemble approach.  
* Returns the final prediction as JSON.

### **API Endpoints**

1. `GET /` \- Serves the web interface.  
2. `POST /predict` \- Accepts an image file, processes it, and returns the prediction.

## **How to Run**

### **Prerequisites**

* Python 3.x  
* Flask  
* TensorFlow/Keras  
* OpenCV

### **Installation**

Clone the repository:  
git clone https://github.com/shubhamprasad318/brain-tumor-detection.git

1. cd brain-tumor-detection  
2. Install dependencies:  
   pip install \-r requirements.txt  
3. Run the Flask application:  
   python app.py  
4. Open `http://127.0.0.1:5000/` in a browser.

