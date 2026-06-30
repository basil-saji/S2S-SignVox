"""
main.py - FastAPI server for Zora sign language translation.

Endpoints:
    POST /translate/landmarks  - Accepts pre-extracted MediaPipe landmarks
    POST /translate/video      - Accepts video file, extracts landmarks server-side
    GET  /health               - Health check
"""

import base64
import io
import logging
import os

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import UniSignDownstreamModel
from tts import TTSEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS", "./weights/model.pth")
TTS_MODEL_PATH = os.getenv("TTS_MODEL", None)  # Path to Piper .onnx model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence threshold to accept prediction

# The 13 target sentences your dataset maps to
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

# ---------------------------------------------------------------------------
# App & Globals
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zora - Sign Language Translation API",
    description="Translates sign language pose sequences to text and speech.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: UniSignDownstreamModel = None
tts_engine: TTSEngine = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global model, tts_engine

    # Load PyTorch model
    logger.info(f"Loading model from {WEIGHTS_PATH} on {DEVICE}")
    model = UniSignDownstreamModel(num_classes=len(SENTENCES)).to(DEVICE)

    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(f"No weights found at {WEIGHTS_PATH}, using random init")

    model.eval()

    # Load TTS engine
    tts_engine = TTSEngine(model_path=TTS_MODEL_PATH)
    logger.info("TTS engine ready")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class LandmarkRequest(BaseModel):
    """Pre-extracted MediaPipe landmarks: list of T frames, each with 180 [x,y,z] points."""
    landmarks: list  # shape: (T, 180, 3)


class TranslationResponse(BaseModel):
    """Translation result with predicted sentence and synthesized audio."""
    sentence: str
    confidence: float
    audio_base64: str


# ---------------------------------------------------------------------------
# Preprocessing helper for inference
# ---------------------------------------------------------------------------
def preprocess_inference_landmarks(landmarks_np: np.ndarray, target_len=60) -> np.ndarray:
    """
    Preprocess landmarks:
      - Slice to first 75 joints (33 Pose + 21 Left Hand + 21 Right Hand)
      - Apply mid-shoulder centering
      - Apply shoulder width scale normalization
      - Uniformly resample to target_len (60 frames)
    """
    # Slice to 75 joints
    T = landmarks_np.shape[0]
    coords = landmarks_np[:, :75, :]  # (T, 75, 3)
    
    preprocessed = []
    prev_mid_shoulder = None
    prev_shoulder_width = None
    
    for t in range(T):
        pose_pts = coords[t, 0:33, :]
        left_pts = coords[t, 33:54, :]
        right_pts = coords[t, 54:75, :]
        
        # Check if pose is detected (not all zeros)
        pose_detected = not np.all(pose_pts == 0.0)
        
        if pose_detected:
            mid_shoulder = (pose_pts[11] + pose_pts[12]) / 2.0
            shoulder_width = np.linalg.norm(pose_pts[11] - pose_pts[12])
            if shoulder_width < 1e-5:
                shoulder_width = 1.0
        else:
            mid_shoulder = prev_mid_shoulder if prev_mid_shoulder is not None else np.zeros(3, dtype=np.float32)
            shoulder_width = prev_shoulder_width if prev_shoulder_width is not None else 1.0
            
        # Center and scale pose
        if pose_detected:
            pose_pts = (pose_pts - mid_shoulder) / shoulder_width
            
        # Center and scale left hand if detected
        if not np.all(left_pts == 0.0):
            left_pts = (left_pts - mid_shoulder) / shoulder_width
        else:
            left_pts = np.zeros((21, 3), dtype=np.float32)
            
        # Center and scale right hand if detected
        if not np.all(right_pts == 0.0):
            right_pts = (right_pts - mid_shoulder) / shoulder_width
        else:
            right_pts = np.zeros((21, 3), dtype=np.float32)
            
        # Assemble frame
        frame_data = np.zeros((75, 3), dtype=np.float32)
        frame_data[0:33] = pose_pts
        frame_data[33:54] = left_pts
        frame_data[54:75] = right_pts
        
        preprocessed.append(frame_data)
        
        if pose_detected:
            prev_mid_shoulder = mid_shoulder
            prev_shoulder_width = shoulder_width
            
    # Resample to target_len (60 frames)
    F = len(preprocessed)
    if F == 0:
        return np.zeros((target_len, 75, 3), dtype=np.float32)
        
    preprocessed = np.array(preprocessed, dtype=np.float32)
    indices = np.linspace(0, F - 1, target_len)
    resampled = np.zeros((target_len, 75, 3), dtype=np.float32)
    
    for t in range(target_len):
        s = indices[t]
        s_low = int(np.floor(s))
        s_high = int(np.ceil(s))
        w = s - s_low
        resampled[t] = (1.0 - w) * preprocessed[s_low] + w * preprocessed[s_high]
        
    return resampled


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def run_inference(landmarks_np: np.ndarray) -> TranslationResponse:
    """
    Run the full inference pipeline:
        Landmarks -> Preprocessing -> Model -> Sentence -> TTS -> Response
    """
    # Preprocess landmarks
    try:
        resampled = preprocess_inference_landmarks(landmarks_np, target_len=60)
    except Exception as e:
        logger.error(f"Preprocessing landmarks failed: {e}")
        return TranslationResponse(
            sentence="Preprocessing error",
            confidence=0.0,
            audio_base64="",
        )

    # Check if any hands were detected (indices 33 to 74 represent left and right hands)
    hands_np = resampled[33:75, :]
    if np.all(hands_np == 0.0):
        return TranslationResponse(
            sentence="No sign detected",
            confidence=0.0,
            audio_base64="",
        )

    # Convert to tensor and permute to (1, 3, 60, 75)
    tensor = torch.tensor(resampled, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_idx = probs.max(dim=1)

    conf = confidence.item()

    # If the confidence is below the threshold, classify it as unrecognized
    if conf < CONFIDENCE_THRESHOLD:
        return TranslationResponse(
            sentence="Unrecognized sign",
            confidence=round(conf, 4),
            audio_base64="",
        )

    sentence = SENTENCES[predicted_idx.item()]

    # Synthesize speech
    try:
        audio_bytes = tts_engine.synthesize(sentence)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        audio_b64 = ""

    return TranslationResponse(
        sentence=sentence,
        confidence=round(conf, 4),
        audio_base64=audio_b64,
    )



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "device": str(DEVICE), "sentences": len(SENTENCES)}


@app.post("/translate/landmarks", response_model=TranslationResponse)
async def translate_landmarks(request: LandmarkRequest):
    """
    Translate pre-extracted MediaPipe Holistic landmarks to a sentence.

    Expects landmarks as a nested list of shape (T, 180, 3).
    """
    try:
        landmarks_np = np.array(request.landmarks, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid landmarks format: {e}")

    if landmarks_np.ndim != 3 or landmarks_np.shape[1] != 180 or landmarks_np.shape[2] != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (T, 180, 3), got {landmarks_np.shape}",
        )

    return run_inference(landmarks_np)


@app.post("/translate/video", response_model=TranslationResponse)
async def translate_video(file: UploadFile = File(...)):
    """
    Upload a video file. The server extracts MediaPipe Holistic
    landmarks frame-by-frame, then runs inference.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="opencv-python and mediapipe are required for video processing",
        )

    # Save uploaded file temporarily in a platform-independent way
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=f"_{file.filename}")
    os.close(fd)
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    # Extract landmarks using MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    all_frames = []

    cap = cv2.VideoCapture(tmp_path)
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            frame_landmarks = []

            # Pose (33 landmarks)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.append([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([[0.0, 0.0, 0.0]] * 33)

            # Left hand (21 landmarks)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.append([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)

            # Right hand (21 landmarks)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.append([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)

            # Face mesh subset (first 105 landmarks to reach 180 total)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    if i >= 105:
                        break
                    frame_landmarks.append([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([[0.0, 0.0, 0.0]] * 105)

            # Ensure exactly 180 landmarks
            frame_landmarks = frame_landmarks[:180]
            while len(frame_landmarks) < 180:
                frame_landmarks.append([0.0, 0.0, 0.0])

            all_frames.append(frame_landmarks)

    cap.release()
    os.remove(tmp_path)

    if not all_frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    landmarks_np = np.array(all_frames, dtype=np.float32)
    return run_inference(landmarks_np)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
