# src/train_keypoint_model_rf_fist_viz.py
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# 获取绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS


def calculate_and_visualize_features(frame, landmarks, draw=True):
    """
    Calculate feature vector from hand landmarks and optionally visualize them on the frame.
    Returns the feature vector.
    """
    if not landmarks:
        return []

    # Get landmark coordinates
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
    palm_center = landmarks[9]  # Using middle MCP as an approximation for palm center

    features = []

    # Convert normalized coordinates to pixel coordinates for drawing
    h, w, c = frame.shape

    def norm_to_pixel(norm_landmark):
        return int(norm_landmark.x * w), int(norm_landmark.y * h)

    pt_thumb_tip = norm_to_pixel(thumb_tip)
    pt_index_tip = norm_to_pixel(index_tip)
    pt_index_mcp = norm_to_pixel(index_mcp)
    pt_middle_tip = norm_to_pixel(middle_tip)
    pt_middle_mcp = norm_to_pixel(middle_mcp)
    pt_ring_tip = norm_to_pixel(ring_tip)
    pt_ring_mcp = norm_to_pixel(ring_mcp)
    pt_pinky_tip = norm_to_pixel(pinky_tip)
    pt_pinky_mcp = norm_to_pixel(pinky_mcp)
    pt_wrist = norm_to_pixel(wrist)
    pt_palm_center = norm_to_pixel(palm_center)

    # 1. Fingertip distances (relative positions) - Visualize as lines
    dist_thumb_index = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    dist_index_middle = np.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)
    dist_middle_ring = np.sqrt((middle_tip.x - ring_tip.x) ** 2 + (middle_tip.y - ring_tip.y) ** 2)
    dist_ring_pinky = np.sqrt((ring_tip.x - pinky_tip.x) ** 2 + (ring_tip.y - pinky_tip.y) ** 2)
    dist_index_pinky = np.sqrt((index_tip.x - pinky_tip.x) ** 2 + (index_tip.y - pinky_tip.y) ** 2)
    dist_thumb_pinky = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)

    features.extend([
        dist_thumb_index, dist_index_middle, dist_middle_ring,
        dist_ring_pinky, dist_index_pinky, dist_thumb_pinky
    ])

    if draw:
        # Draw fingertip distance lines
        cv2.line(frame, pt_thumb_tip, pt_index_tip, (255, 0, 0), 2)  # Blue
        cv2.line(frame, pt_index_tip, pt_middle_tip, (255, 0, 0), 2)
        cv2.line(frame, pt_middle_tip, pt_ring_tip, (255, 0, 0), 2)
        cv2.line(frame, pt_ring_tip, pt_pinky_tip, (255, 0, 0), 2)
        cv2.line(frame, pt_index_tip, pt_pinky_tip, (255, 0, 0), 2)
        cv2.line(frame, pt_thumb_tip, pt_pinky_tip, (255, 0, 0), 2)

    # 2. Distance from fingertips to palm center - Visualize as lines
    dist_index_to_palm = np.sqrt((index_tip.x - palm_center.x) ** 2 + (index_tip.y - palm_center.y) ** 2)
    dist_middle_to_palm = np.sqrt((middle_tip.x - palm_center.x) ** 2 + (middle_tip.y - palm_center.y) ** 2)
    dist_ring_to_palm = np.sqrt((ring_tip.x - palm_center.x) ** 2 + (ring_tip.y - palm_center.y) ** 2)
    dist_pinky_to_palm = np.sqrt((pinky_tip.x - palm_center.x) ** 2 + (pinky_tip.y - palm_center.y) ** 2)

    features.extend([dist_index_to_palm, dist_middle_to_palm, dist_ring_to_palm, dist_pinky_to_palm])

    if draw:
        # Draw fingertip to palm lines
        cv2.line(frame, pt_index_tip, pt_palm_center, (0, 255, 0), 2)  # Green
        cv2.line(frame, pt_middle_tip, pt_palm_center, (0, 255, 0), 2)
        cv2.line(frame, pt_ring_tip, pt_palm_center, (0, 255, 0), 2)
        cv2.line(frame, pt_pinky_tip, pt_palm_center, (0, 255, 0), 2)

    # 3. Distance from fingertips to their corresponding MCP joints (indicating how bent they are)
    dist_index_tip_to_mcp = np.sqrt((index_tip.x - index_mcp.x) ** 2 + (index_tip.y - index_mcp.y) ** 2)
    dist_middle_tip_to_mcp = np.sqrt((middle_tip.x - middle_mcp.x) ** 2 + (middle_tip.y - middle_mcp.y) ** 2)
    dist_ring_tip_to_mcp = np.sqrt((ring_tip.x - ring_mcp.x) ** 2 + (ring_tip.y - ring_mcp.y) ** 2)
    dist_pinky_tip_to_mcp = np.sqrt((pinky_tip.x - pinky_mcp.x) ** 2 + (pinky_tip.y - pinky_mcp.y) ** 2)
    # Thumb is different, its tip moves towards the base of other fingers when making a fist
    dist_thumb_tip_to_wrist = np.sqrt((thumb_tip.x - wrist.x) ** 2 + (thumb_tip.y - wrist.y) ** 2)

    features.extend([dist_index_tip_to_mcp, dist_middle_tip_to_mcp, dist_ring_tip_to_mcp, dist_pinky_tip_to_mcp,
                     dist_thumb_tip_to_wrist])

    if draw:
        # Draw fingertip to MCP lines
        cv2.line(frame, pt_index_tip, pt_index_mcp, (0, 0, 255), 2)  # Red
        cv2.line(frame, pt_middle_tip, pt_middle_mcp, (0, 0, 255), 2)
        cv2.line(frame, pt_ring_tip, pt_ring_mcp, (0, 0, 255), 2)
        cv2.line(frame, pt_pinky_tip, pt_pinky_mcp, (0, 0, 255), 2)
        # Draw thumb tip to wrist line
        cv2.line(frame, pt_thumb_tip, pt_wrist, (0, 255, 255), 2)  # Yellow

    return features


# --- Data Collection for Random Forest Model with 'fist' gesture and Visualization ---
data = []
labels = []

# Updated gesture labels
GESTURE_LABELS = ['move', 'click_left', 'click_right', 'scroll_up', 'fist']
sample_counts = {label: 0 for label in GESTURE_LABELS}

cap = cv2.VideoCapture(0)

print("Collecting Keypoint Data for Random Forest Training (with 'fist') and Visualization...")
print("Press Key to Label:")
for i, label in enumerate(GESTURE_LABELS):
    print(f"  '{i}' for '{label}'")
print("Press 'q' to quit and start training.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb_frame)

    current_label = None
    key = cv2.waitKey(1) & 0xFF

    for i, label in enumerate(GESTURE_LABELS):
        if key == ord(str(i)):
            current_label = label
            break

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Calculate features AND visualize them on the frame
            features = calculate_and_visualize_features(frame, hand_landmarks.landmark, draw=True)

            if current_label and len(features) > 0:  # Ensure features were calculated and a label was pressed
                data.append(features)
                labels.append(current_label)
                sample_counts[current_label] += 1
                print(f"Collected sample for '{current_label}'. Total: {sample_counts[current_label]}")

        # Draw standard MediaPipe landmarks on top of our custom visualizations
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands_connections)

    else:
        # If no hand is detected, still show the frame (it will be blank)
        pass

    # Display counts
    y_pos = 30
    for label, count in sample_counts.items():
        color = (0, 255, 0) if count >= 200 else (0, 255, 255)
        cv2.putText(frame, f'{label}: {count}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 30

    cv2.imshow('Collect Keypoint Data (RF - Fist) with Viz', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Train the Random Forest model ---
if len(data) == 0 or len(labels) == 0:
    print("No keypoint data collected!")
    exit()

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Note: StandardScaler is often less critical for tree-based models like Random Forest,
# but it doesn't hurt and maintains consistency if you switch back to other algorithms later.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler
model_save_path = "../models/keypoint_rf_model_fist.pkl"
scaler_save_path = "../models/keypoint_feature_scaler_fist.pkl"

joblib.dump(rf_model, model_save_path)
joblib.dump(scaler, scaler_save_path)

print(f"Random Forest Model and Scaler saved to {model_save_path} and {scaler_save_path}")