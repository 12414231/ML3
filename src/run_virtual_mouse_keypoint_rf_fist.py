# src/run_virtual_mouse_keypoint_rf_fist_click_once.py
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import joblib
import time

# Load the trained Random Forest model and scaler (the new ones with 'fist')
model_path = "../models/keypoint_rf_model_fist.pkl"
scaler_path = "../models/keypoint_feature_scaler_fist.pkl"

try:
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    print("Random Forest model (with 'fist') and scaler loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: Could not find model file {model_path} or scaler file {scaler_path}. Did you run train_keypoint_model_rf_fist.py?")
    exit()


def calculate_features(landmarks):
    """
    Same function used during training.
    """
    if not landmarks:
        return []

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]
    wrist = landmarks[0]
    palm_center = landmarks[9]

    features = []

    dist_thumb_index = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    dist_index_middle = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5
    dist_middle_ring = ((middle_tip.x - ring_tip.x) ** 2 + (middle_tip.y - ring_tip.y) ** 2) ** 0.5
    dist_ring_pinky = ((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2) ** 0.5
    dist_index_pinky = ((index_tip.x - pinky_tip.x) ** 2 + (index_tip.y - pinky_tip.y) ** 2) ** 0.5
    dist_thumb_pinky = ((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2) ** 0.5

    features.extend([
        dist_thumb_index, dist_index_middle, dist_middle_ring,
        dist_ring_pinky, dist_index_pinky, dist_thumb_pinky
    ])

    # Corrected line: Calculate distance from index finger tip to palm center
    dist_index_to_palm = ((index_tip.x - palm_center.x) ** 2 + (index_tip.y - palm_center.y) ** 2) ** 0.5
    dist_middle_to_palm = ((middle_tip.x - palm_center.x) ** 2 + (middle_tip.y - palm_center.y) ** 2) ** 0.5
    dist_ring_to_palm = ((ring_tip.x - palm_center.x) ** 2 + (ring_tip.y - palm_center.y) ** 2) ** 0.5
    dist_pinky_to_palm = ((pinky_tip.x - palm_center.x) ** 2 + (pinky_tip.y - palm_center.y) ** 2) ** 0.5

    features.extend([dist_index_to_palm, dist_middle_to_palm, dist_ring_to_palm, dist_pinky_to_palm])

    # Finger bend features
    dist_index_tip_to_mcp = ((index_tip.x - index_mcp.x) ** 2 + (index_tip.y - index_mcp.y) ** 2) ** 0.5
    dist_middle_tip_to_mcp = ((middle_tip.x - middle_mcp.x) ** 2 + (middle_tip.y - middle_mcp.y) ** 2) ** 0.5
    dist_ring_tip_to_mcp = ((ring_tip.x - ring_mcp.x) ** 2 + (ring_tip.y - ring_mcp.y) ** 2) ** 0.5
    dist_pinky_tip_to_mcp = ((pinky_tip.x - pinky_mcp.x) ** 2 + (pinky_tip.y - pinky_mcp.y) ** 2) ** 0.5
    dist_thumb_tip_to_wrist = ((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2) ** 0.5

    features.extend([dist_index_tip_to_mcp, dist_middle_tip_to_mcp, dist_ring_tip_to_mcp, dist_pinky_tip_to_mcp,
                     dist_thumb_tip_to_wrist])

    return features


def run_virtual_mouse_keypoint_rf_fist_click_once(model, scaler, camera_index=0, confidence_threshold=0.7):
    """
    Run the virtual mouse using MediaPipe keypoints and the trained Random Forest model (with 'fist').
    Modified to execute clicks only once per gesture appearance.
    """
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    screen_width, screen_height = pyautogui.size()
    prev_x, prev_y = 0, 0
    smoothing_factor = 3  # Adjusted from previous example
    last_gesture_time = time.time()
    gesture_cooldown = 0.2

    # State variable to track if a click gesture is currently active
    active_click_gesture = None

    # Sensitivity for mouse movement
    sensitivity = 1.3  # Adjust this value: > 1.0 makes it faster, < 1.0 makes it slower

    print("Keypoint-based Virtual Mouse (RF - Fist - Single Click) started. Press 'q' to quit.")
    print(
        "Instructions: 'move' to move mouse, 'click_left'/'click_right' for single clicks, 'scroll_up'/'fist' to scroll.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate features from landmarks
                features = calculate_features(hand_landmarks.landmark)

                if len(features) > 0:
                    # Prepare for prediction
                    features_array = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features_array)

                    # Predict gesture using the Random Forest model
                    prediction = model.predict(features_scaled)[0]
                    probabilities = model.predict_proba(features_scaled)[0]
                    max_prob_idx = np.argmax(probabilities)
                    confidence = probabilities[max_prob_idx]

                    # Draw landmarks for visualization
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks,
                                                              mp.solutions.hands.HAND_CONNECTIONS)

                    # --- Control Logic with Single Click Execution ---
                    if confidence > confidence_threshold:
                        # Map index finger tip position to screen for 'move', correcting for mirror and applying sensitivity
                        index_finger_tip = hand_landmarks.landmark[8]

                        # Correct the mirrored x-axis and apply sensitivity
                        x_norm_corrected = 1 - index_finger_tip.x
                        y_norm = index_finger_tip.y

                        # Apply sensitivity to the raw normalized values before mapping to screen size
                        delta_x = (x_norm_corrected - 0.5) * sensitivity
                        delta_y = (y_norm - 0.5) * sensitivity

                        # Map to screen coordinates based on the scaled deltas
                        target_x = screen_width * (0.5 + delta_x)
                        target_y = screen_height * (0.5 + delta_y)

                        if prediction == 'move':
                            # Always execute move
                            smooth_x = int((prev_x * (smoothing_factor - 1) + target_x) / smoothing_factor)
                            smooth_y = int((prev_y * (smoothing_factor - 1) + target_y) / smoothing_factor)
                            pyautogui.moveTo(smooth_x, smooth_y)
                            prev_x, prev_y = smooth_x, smooth_y
                            # Reset click state if we switch to move
                            active_click_gesture = None
                            # print(f"Moving... ({smooth_x}, {smooth_y})") # Uncomment if needed for debugging

                        elif prediction in ['click_left', 'click_right']:
                            # Execute click only if it's a *new* gesture
                            if prediction != active_click_gesture:
                                if prediction == 'click_left':
                                    pyautogui.click(button='left')
                                    print("Left Click!")
                                elif prediction == 'click_right':
                                    pyautogui.click(button='right')
                                    print("Right Click!")
                                # Update the active click gesture
                                active_click_gesture = prediction
                                # Optional: Add a small delay if needed, though cooldown is handled by gesture change
                                # time.sleep(0.05)

                        elif prediction == 'scroll_up' and (current_time - last_gesture_time) > gesture_cooldown:
                            pyautogui.scroll(36)  # Increased scroll amount
                            print("Scroll Up!")
                            last_gesture_time = current_time
                            # Reset click state if we switch to scroll
                            active_click_gesture = None

                        elif prediction == 'fist' and (current_time - last_gesture_time) > gesture_cooldown:
                            pyautogui.scroll(-36)  # Increased scroll amount
                            print("Scroll Down (Fist)!")
                            last_gesture_time = current_time
                            # Reset click state if we switch to scroll
                            active_click_gesture = None

                        # Display prediction and active click status on frame
                        status_text = f'Gesture: {prediction} ({confidence:.2f})'
                        if active_click_gesture:
                            status_text += f" [ACTIVE: {active_click_gesture}]"
                        cv2.putText(frame, status_text,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    else:
                        # Display low confidence
                        cv2.putText(frame, f'Low Confidence: {confidence:.2f}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Break after processing first hand
                break

        else:
            # No hand detected, reset smoothing and active click state
            prev_x, prev_y = 0, 0
            active_click_gesture = None  # Reset click state when hand disappears
            cv2.putText(frame, 'No Hand Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Keypoint Virtual Mouse (Single Click)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Keypoint-based Virtual Mouse (Single Click) stopped.")


if __name__ == "__main__":
    run_virtual_mouse_keypoint_rf_fist_click_once(loaded_model, loaded_scaler)