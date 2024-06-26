from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

app = Flask(__name__)

# Load your Keras model
model = tf.keras.models.load_model('action.keras')
actions = ['hello', 'thanks', 'home']  # Adjust according to your label map

# Initialize variables for sequence, sentence, predictions, thresholds, etc.
sequence = []
sentence = []
predictions = deque(maxlen=10)  # Store the last 10 predictions
threshold = 0.7
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture and Mediapipe holistic model
cap = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Make the image read-only to improve performance
    results = model.process(image)  # Perform detection with the provided model
    image.flags.writeable = True  # Make the image writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmark(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, 
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def gen_frames():
    global sequence, sentence, predictions

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            draw_styled_landmark(image, results)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-10:]

            # Perform prediction
            if len(sequence) == 10:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                print(actions[np.argmax(res)])

                # Use majority voting over the last 10 predictions
                if len(predictions) == predictions.maxlen:
                    most_common_action = np.bincount(predictions).argmax()
                    if res[most_common_action] > threshold:
                        if len(sentence) == 0 or actions[most_common_action] != sentence[-1]:
                            sentence.append(actions[most_common_action])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Visualize probabilities if prob_viz function is defined
                if 'prob_viz' in globals():
                    image = prob_viz(res, actions, image, colors)

            # Display the action on the frame
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
