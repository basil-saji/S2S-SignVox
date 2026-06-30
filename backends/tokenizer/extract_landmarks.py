"""
extract_landmarks.py - Extract MediaPipe Holistic landmarks from raw videos locally

This script walks through a folder of raw videos (e.g., ../raw_videos), 
extracts landmarks using MediaPipe Holistic, and saves them as .npy files

Usage:
    python extract_landmarks.py
"""

import os
import cv2
import mediapipe as mp
import numpy as np

# Config
RAW_VIDEOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "raw_videos"))
V2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "V2"))

SENTENCES = [
    "Can you help me?",
    "Can I help you",
    "Can you wait?",
    "Can you repeat?",
    "Can you see me?",
    "Please help me",
    "Please wait",
    "Please repeat",
    "I need help",
    "I understand",
    "I don't understand",
    "Do you understand?",
    "Do you need help?",
]

def get_sentence_class_id(folder_name):
    normalized_folder = folder_name.lower().replace("?", "").replace("'", "").replace("-", " ").strip()
    for idx, sentence in enumerate(SENTENCES):
        normalized_sentence = sentence.lower().replace("?", "").replace("'", "").replace("-", " ").strip()
        if normalized_folder in normalized_sentence or normalized_sentence in normalized_folder:
            return idx
    # Check for common variants
    if "not understand" in normalized_folder:
        return 10  # Map "I NOT UNDERSTAND" to "I don't understand"
    return -1

mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            fl = []
            # Pose (33 landmarks)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    fl.append([lm.x, lm.y, lm.z])
            else:
                fl.extend([[0.0, 0.0, 0.0]] * 33)
                
            # Left Hand (21 landmarks)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    fl.append([lm.x, lm.y, lm.z])
            else:
                fl.extend([[0.0, 0.0, 0.0]] * 21)
                
            # Right Hand (21 landmarks)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    fl.append([lm.x, lm.y, lm.z])
            else:
                fl.extend([[0.0, 0.0, 0.0]] * 21)
                
            # Face Mesh subset (105 landmarks)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    if i >= 105:
                        break
                    fl.append([lm.x, lm.y, lm.z])
            else:
                fl.extend([[0.0, 0.0, 0.0]] * 105)
                
            fl = fl[:180]
            while len(fl) < 180:
                fl.append([0.0, 0.0, 0.0])
            frames_landmarks.append(fl)
            
    cap.release()
    return np.array(frames_landmarks, dtype=np.float32)

def main():
    print("==================================================")
    print("        Zora Local Landmark Extractor             ")
    print("==================================================")
    
    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"Error: Raw videos directory not found at: {RAW_VIDEOS_DIR}")
        print("Please download the videos from Google Drive and place them there.")
        return
        
    os.makedirs(V2_DIR, exist_ok=True)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    extracted_count = 0
    skipped_count = 0
    
    print(f"Scanning {RAW_VIDEOS_DIR} for videos...")
    
    for root, dirs, files in os.walk(RAW_VIDEOS_DIR):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                phrase_folder = os.path.basename(root)
                parent_dir = os.path.dirname(root)
                signer_folder = os.path.basename(parent_dir)
                
                class_id = get_sentence_class_id(phrase_folder)
                if class_id == -1:
                    class_id = get_sentence_class_id(signer_folder)
                    
                if class_id != -1:
                    out_name = f"class{class_id}_{signer_folder}_{os.path.splitext(file)[0]}.npy"
                    out_path = os.path.join(V2_DIR, out_name)
                    
                    if not os.path.exists(out_path):
                        print(f"Extracting: {signer_folder} -> {phrase_folder} -> {file}")
                        try:
                            landmarks = extract_landmarks_from_video(video_path)
                            if len(landmarks) > 0:
                                np.save(out_path, landmarks)
                                extracted_count += 1
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                    else:
                        print(f"Skipping (already exists): {out_name}")
                else:
                    print(f"Skipping folder (unmapped phrase): '{phrase_folder}' or signer: '{signer_folder}' for video: {file}")
                    skipped_count += 1
                    
    print(f"\nLandmark extraction complete!")
    print(f"Successfully extracted: {extracted_count} videos.")
    print(f"Skipped unmapped videos: {skipped_count}.")
    print(f"Landmark files are stored in: {V2_DIR}")

if __name__ == "__main__":
    main()
