import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from datetime import datetime

# Path to save captured images (where Flask was saving)
SAVE_PATH = r"D:\Python\Health_AI\static\temp_images"

# Ensure the folder exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Load the pre-trained model (ensure it's trained on RGB (48, 48, 3) images)
model = tf.keras.models.load_model(r"D:\Python\Health_AI\training_2\best_emotion_model.keras")

# Define your MCQs and capture responses
mcq_questions = [
    "How are you feeling today?",
    "Do you feel anxious often?",
    "Do you enjoy your daily activities?",
    "Do you find it difficult to focus?",
    "Do you feel emotionally stable?",
    "Are you satisfied with your work/life balance?",
    "Do you feel optimistic about the future?",
    "Do you feel sad or depressed frequently?",
    "Do you find it hard to relax?",
    "Do you feel happy and content with life?"
]

# Placeholder for storing the user's answers
responses = {}

# Create MCQ form using a more structured layout
st.markdown("<h1 style='text-align: center; color: blue;'>Mental Health Check Using MCQs and Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please answer the following questions and capture your emotions at key moments.</p>", unsafe_allow_html=True)

# MCQ questions loop
for i, question in enumerate(mcq_questions):
    responses[question] = st.radio(f"<b>{i+1}. {question}</b>", ("Yes", "No", "Sometimes"), key=f"q_{i}")

# Placeholder for image capture
captured_images = []

# Function to save captured image in RGB format
def save_image(image_data, question_number):
    # Save the image in the specified folder with a timestamp
    filename = f"captured_image_q{question_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image_path = os.path.join(SAVE_PATH, filename)
    
    # Convert image to RGB (since model was trained on RGB) and save
    image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (48, 48))  # Resize to (48, 48)
    cv2.imwrite(image_path, resized_image)
    
    return image_path

# Real-time image capture logic using webcam (OpenCV)
def capture_image():
    cam = cv2.VideoCapture(0)  # 0 is typically the default camera
    ret, frame = cam.read()
    if ret:
        st.image(frame, channels="BGR")  # Display the captured image in the app
        return frame
    else:
        st.error("Failed to capture image.")
        return None

# Image capture at the 1st, 5th, and 10th questions
if st.button("Capture Image (1st Question)"):
    frame = capture_image()
    if frame is not None:
        image_path = save_image(frame, 1)
        st.write(f"Image captured at 1st question and saved to {image_path}")
        captured_images.append(image_path)

if st.button("Capture Image (5th Question)"):
    frame = capture_image()
    if frame is not None:
        image_path = save_image(frame, 5)
        st.write(f"Image captured at 5th question and saved to {image_path}")
        captured_images.append(image_path)

if st.button("Capture Image (10th Question)"):
    frame = capture_image()
    if frame is not None:
        image_path = save_image(frame, 10)
        st.write(f"Image captured at 10th question and saved to {image_path}")
        captured_images.append(image_path)

# Predict emotion from images (RGB, 48x48, model input size)
def predict_emotion(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))  # Resize to the model input size
    img = img.astype("float32") / 255  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand to batch size
    
    predictions = model.predict(img)[0]
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = emotion_labels[np.argmax(predictions)]
    
    return predicted_emotion, predictions

# Predict final emotion based on three images
def predict_final_emotion(image_paths, model):
    emotions = []
    predictions_list = []

    for image_path in image_paths:
        emotion, predictions = predict_emotion(image_path, model)
        emotions.append(emotion)
        predictions_list.append(predictions)

    # Majority voting for final emotion
    final_emotion = max(set(emotions), key=emotions.count)
    final_predictions = np.mean(predictions_list, axis=0)  # Average confidence for bar chart
    return final_emotion, final_predictions

# Combine MCQ and emotion predictions
def combine_assessments(predicted_emotion, mental_state):
    final_assessment = ""

    if "significant emotional challenges" in mental_state:
        if predicted_emotion in ["Sad", "Angry", "Fear"]:
            final_assessment = "You're facing significant challenges and negative emotions. Seek support."
        elif predicted_emotion == "Happy":
            final_assessment = "You're facing challenges but feeling happy. Stay positive!"
        else:
            final_assessment = "You're facing challenges, but emotions are neutral. Reflect on your feelings."
    elif "emotional challenges" in mental_state:
        if predicted_emotion in ["Sad", "Angry"]:
            final_assessment = "You have emotional challenges and negative emotions. Reach out for support."
        elif predicted_emotion == "Happy":
            final_assessment = "You have challenges, but you're happy. Focus on positivity!"
        else:
            final_assessment = "You have challenges, but emotions are neutral. Stay balanced."
    else:
        final_assessment = f"You're in a good mental state, and your emotion is {predicted_emotion}. Keep up the positive mindset!"

    return final_assessment

# Combine the final MCQ and emotion results
if st.button("Submit and Get Final Assessment"):
    # Analyze MCQ responses to determine mental state
    mental_state = "good mental state"
    negative_responses = [ans for ans in responses.values() if ans == "Yes"]

    if len(negative_responses) > 5:
        mental_state = "significant emotional challenges"
    elif len(negative_responses) > 2:
        mental_state = "emotional challenges"

    # Ensure all images are captured
    if len(captured_images) == 3:
        final_emotion, emotion_confidences = predict_final_emotion(captured_images, model)
        final_assessment = combine_assessments(final_emotion, mental_state)
        st.subheader("Final Assessment")
        st.write(final_assessment)

        # Display emotion confidence levels as a bar chart
        st.subheader("Emotion Confidence Levels")
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        fig, ax = plt.subplots()
        ax.bar(emotion_labels, emotion_confidences)
        st.pyplot(fig)
    else:
        st.error("Please capture images at all three stages.")
