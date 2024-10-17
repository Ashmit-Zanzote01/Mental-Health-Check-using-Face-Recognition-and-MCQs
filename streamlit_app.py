import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

# Load the pre-trained model
@st.cache_resource
def load_prediction_model():
    return tf.keras.models.load_model('best_emotion_model.keras')  # Ensure this model is in your GitHub repo

model = load_prediction_model()

# Function to predict emotion from an image
def predict_emotion(image):
    if model is None:
        return "Model not loaded"

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    emotion_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    return emotion_map.get(int(predicted_class[0]), "Unknown"), prediction[0]

# Define MCQs and capture responses
mcq_questions = [
    {"question": "How do you feel when you wake up in the morning?", "options": ["A. Excited and motivated", "B. Anxious or stressed", "C. Frustrated or irritated", "D. Indifferent or neutral"]},
    {"question": "How do you typically react to an unexpected problem or challenge?", "options": ["A. I feel overwhelmed and shut down", "B. I feel frustrated and lose my temper", "C. I stay calm and look for a solution", "D. I feel excited by the challenge"]},
    {"question": "How often do you feel sad or down?", "options": ["A. Almost every day", "B. A few times a week", "C. Rarely", "D. Never"]},
    {"question": "Do you find it easy to express your emotions?", "options": ["A. Yes, always", "B. Sometimes", "C. Rarely", "D. No, never"]},
    {"question": "How do you handle stress?", "options": ["A. I manage it well", "B. I sometimes struggle", "C. I often feel overwhelmed", "D. I ignore it"]},
    {"question": "How often do you feel anxious?", "options": ["A. Very often", "B. Occasionally", "C. Rarely", "D. Never"]},
    {"question": "Do you enjoy social interactions?", "options": ["A. Yes, I love it", "B. Sometimes", "C. Not really", "D. No, I avoid them"]},
    {"question": "How often do you feel bored or unfulfilled?", "options": ["A. Often", "B. Sometimes", "C. Rarely", "D. Never"]},
    {"question": "How do you feel about your daily activities?", "options": ["A. Very positive", "B. Somewhat positive", "C. Neutral", "D. Negative"]},
    {"question": "How would you rate your overall mood today?", "options": ["A. Excellent", "B. Good", "C. Fair", "D. Poor"]}
]

# Placeholder for storing the user's answers
responses = {}

# Create MCQ form
st.markdown("<h1 style='text-align: center; color: blue;'>Mental Health Check Using MCQs and Emotion Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Please answer the following questions and capture your emotion at the end.</p>", unsafe_allow_html=True)

# MCQ questions loop
for i, item in enumerate(mcq_questions):
    responses[item["question"]] = st.radio(f"<b>{i + 1}. {item['question']}</b>", item["options"], key=f"q_{i}")

# Placeholder for the captured image
captured_image = None

# Real-time image capture logic using webcam
def capture_image():
    img_file_buffer = st.camera_input("Capture your emotion at the end")
    if img_file_buffer is not None:
        # Decode the image buffer into an RGB format image
        image = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), cv2.IMREAD_COLOR)
        return image
    return None

# Capture the image after answering all questions
if st.button("Capture Final Image"):
    captured_image = capture_image()
    if captured_image is not None:
        st.image(captured_image, channels="RGB")  # Display the captured image in the app

# Assess mental state based on responses
def assess_mental_state(responses):
    score = sum(1 for key in responses if responses[key].startswith("C") or responses[key].startswith("D"))
    if score >= 5:
        return "You might be experiencing significant emotional challenges. It's important to talk to someone."
    elif score >= 3:
        return "You might have some emotional challenges. Consider reaching out for support."
    else:
        return "You seem to be in a good mental state! Keep up the positive mindset."

# Combine the final MCQ and emotion results
if st.button("Submit and Get Final Assessment"):
    mental_state = assess_mental_state(responses)

    # Ensure an image is captured
    if captured_image is not None:
        predicted_emotion, emotion_confidences = predict_emotion(captured_image)
        
        final_assessment = combine_assessments(predicted_emotion, mental_state)
        
        st.subheader("Final Assessment")
        st.write(final_assessment)

        # Display emotion confidence levels as a bar chart
        st.subheader("Emotion Confidence Levels")
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        fig, ax = plt.subplots()
        ax.bar(emotion_labels, emotion_confidences)
        st.pyplot(fig)
    else:
        st.error("Please capture your emotion image at the end.")

# Function to combine assessments
def combine_assessments(predicted_emotion, mental_state):
    final_assessment = ""

    if "significant emotional challenges" in mental_state:
        if predicted_emotion in ["Sad", "Angry", "Fear"]:
            final_assessment = "It seems you're facing significant challenges and also feeling negative emotions. It's crucial to seek support."
        elif predicted_emotion == "Happy":
            final_assessment = "You might be experiencing challenges, but it's great to see you're feeling happy. Keep engaging in positive activities!"
        else:
            final_assessment = "You're experiencing challenges, but your emotions are neutral. Consider reflecting on your feelings."
    elif "emotional challenges" in mental_state:
        if predicted_emotion in ["Sad", "Angry"]:
            final_assessment = "You might have emotional challenges and are feeling negative emotions. Consider reaching out for support."
        elif predicted_emotion == "Happy":
            final_assessment = "You might have challenges, but it's uplifting to see you're feeling happy. Focus on positive actions!"
        else:
            final_assessment = "You have some challenges, but your emotions are neutral. Try to stay positive."
    else:
        final_assessment = f"You seem to be in a good mental state, and your emotion is {predicted_emotion}. Keep up the positive mindset!"

    return final_assessment
