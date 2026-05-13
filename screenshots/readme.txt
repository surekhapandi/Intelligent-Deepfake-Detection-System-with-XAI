PROJECT TITLE:
Intelligent Human Face Deepfake Detection System using Explainable AI

DEVELOPED BY:
Pandi Surekha
24844510045

DESCRIPTION:
This project detects whether a video is real or fake using deep learning and explainable AI techniques. It uses InceptionV3 for feature extraction and a sequence model for prediction. The system also provides explainability using Grad-CAM and SHAP.

TECHNOLOGIES USED:
 Python 
 TensorFlow & Keras
 OpenCV
 Streamlit
 NumPy, Pandas
 VS Code

HOW TO RUN THE PROJECT:

1. Open the project folder in VS Code (or any editor).

2. Open the terminal (Command Prompt).

3. (Optional but recommended) Create a virtual environment:
   python -m venv venv

4. Activate the virtual environment:

    For Windows:
     venv\Scripts\activate

5. Install required libraries:
   pip install -r requirements.txt
   (or install manually: TensorFlow, OpenCV, Streamlit, etc.)

6. Run the application:
   streamlit run app.py

7. The application will open in your browser.
   Upload a video and click “Detect” to view results.


FILES INCLUDED:

 app.py (Main application file)
 Model file (inceptionNet_model.h5)
 README file
 Project documentation
 Sample videos (if included)

OUTPUT:

 Prediction (Real/Fake with confidence)
 SHAP values graph
 Frame importance graph
 Grad-CAM heatmaps

NOTE:
Ensure all required libraries are installed before running the project.
