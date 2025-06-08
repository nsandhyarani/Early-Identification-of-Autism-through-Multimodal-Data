# Early-Identification-of-Autism-through-Multimodal-Data

# Overview
This project proposes a non-invasive system for the early identification of Autism Spectrum Disorder (ASD) by integrating image and behavioral data. Traditional diagnostic methods are often time-consuming, subjective, and reliant on expert interpretation. This system addresses these limitations by utilizing thermal-like image features and psychological survey data processed via deep learning models, specifically Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN). The model is designed for rapid, scalable, and accurate screening of ASD, particularly in resource-limited environments.
# Features
Multimodal Inputs: Combines facial image data and behavioral survey responses.  
Transfer Learning with VGG16: Efficient feature extraction from facial images.  
Artificial Neural Network (ANN): Processes survey-based psychological data.  
Fusion Model: Merges outputs from CNN and ANN for a more comprehensive diagnosis.  
Streamlit Interface: Provides real-time prediction and diagnostic reports.  
# Methodology
• Data Collection: Facial images sourced from Kaggle and behavioral data from UCL repositories.  
• Preprocessing: Image resizing, normalization, and augmentation; survey encoding and cleaning.  
• CNN Architecture: VGG16 base with custom classification layers using softmax.    
• ANN Architecture: Feed-forward neural network to handle survey data.  
• Multimodal Fusion: Combines extracted features before classification.  
• Evaluation: Accuracy, Precision, Recall, and F1-score metrics were used to validate the model.  
# Output
The integrated system successfully identifies whether a subject is likely to have ASD based on facial expressions and survey data. Predictions are displayed through a user-friendly interface that includes confidence scores and diagnostic summaries, making it accessible even for non-experts. The use of a multimodal approach ensures higher diagnostic reliability and broader applicability.
[![Watch the demo]("C:\Users\Sandhya\Desktop\Projects\Mini Project(3-2)\M3_EXECUTION-vedio.mp4")

# Results
Training Accuracy: 92%  
Validation Accuracy: 84%  
Non-ASD Precision: 0.88, Recall: 0.90, F1 Score: 0.89  
ASD Precision: 0.78, Recall: 0.75, F1 Score: 0.76  
Multimodal fusion improved performance compared to single-input models.  
# Conclusion
This project presents an automated and non-invasive approach to ASD diagnosis using multimodal data. The integration of CNN and ANN models enhances accuracy and minimizes subjectivity in diagnosis. Future enhancements will include expansion to audio and video modalities, mobile deployment, and real-time predictive systems to ensure broader impact and accessibility.
