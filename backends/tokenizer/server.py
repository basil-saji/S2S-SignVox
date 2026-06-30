"""
server.py - Zora Sign Language Translation Server (Recorded Video)

Run this file to start the FastAPI server on http://localhost:8000

Usage:
    python server.py

The server will:
    1. Load vocab and phrase mappings from ./weights/
    2. Build the MeetixSeq2Seq model and load trained weights
    3. Load the Fingerspelling MLP model from ./fingerspell_models/
    4. Start a FastAPI server on port 8000
    5. Accept uploaded video files and return translations + TTS audio
"""

import base64
import json
import logging
import os
import tempfile

import numpy as np
import torch
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import UniSignEncoderOnly, MeetixSeq2Seq, mediapipe_to_unisign
from tts import TTSEngine
from features_alphabet import featurize_pose
import keras
from collections import deque

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.json")
PHRASE_MAP_PATH = os.path.join(WEIGHTS_DIR, "phrase_to_tokens.json")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_meetix_seq2seq.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.50

# Token-tuple -> human-readable phrase lookup
TOKEN_SEQ_TO_PHRASE = {
    ("CAN", "I", "HELP_YOU", "YOU"): "Can I help you?",
    ("CAN", "YOU", "HELP_ME"): "Can you help me?",
    ("CAN", "YOU", "REPEAT"): "Can you repeat?",
    ("CAN", "YOU", "SEE", "ME"): "Can you see me?",
    ("CAN", "YOU", "WAIT"): "Can you wait?",
    ("I", "NEED", "HELP_ME"): "I need help",
    ("I", "NOT", "UNDERSTAND"): "I don't understand",
    ("I", "UNDERSTAND"): "I understand",
    ("PLEASE", "HELP_ME"): "Please help me",
    ("PLEASE", "REPEAT"): "Please repeat",
    ("PLEASE", "WAIT"): "Please wait",
    ("YOU", "NEED", "HELP_YOU"): "Do you need help?",
    ("YOU", "UNDERSTAND"): "Do you understand?",
}

# ---------------------------------------------------------------------------
# App & Globals
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Zora - Sign Language Translation API",
    description="Translates sign language pose sequences to text and speech (Seq2Seq).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: MeetixSeq2Seq = None
tts_engine: TTSEngine = None
VOCAB: dict = None  # token_str -> id
ID_TO_TOKEN: dict = None  # id -> token_str
SOS_ID: int = None
EOS_ID: int = None

fingerspell_model = None
fingerspell_labels = None



# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global model, tts_engine, VOCAB, ID_TO_TOKEN, SOS_ID, EOS_ID, fingerspell_model, fingerspell_labels

    logger.info(f"Device: {DEVICE}")

    # Load vocabulary
    if not os.path.exists(VOCAB_PATH):
        logger.error(f"Vocab file not found at {VOCAB_PATH}")
        raise RuntimeError(f"Vocab file not found at {VOCAB_PATH}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    # vocab.json has: {"VOCAB": [...], "token_to_id": {...}, "id_to_token": {...}}
    VOCAB = vocab_data["token_to_id"]
    ID_TO_TOKEN = {int(k): v for k, v in vocab_data["id_to_token"].items()}
    SOS_ID = VOCAB["<SOS>"]
    EOS_ID = VOCAB["<EOS>"]
    vocab_size = len(VOCAB)

    logger.info(f"Vocab loaded: {vocab_size} tokens, SOS={SOS_ID}, EOS={EOS_ID}")

    # Build model
    encoder = UniSignEncoderOnly()
    model = MeetixSeq2Seq(encoder, vocab_size=vocab_size).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading trained model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights loaded successfully")
    else:
        logger.warning(f"No weights found at {MODEL_PATH}")

    model.eval()

    # Load TTS engine (gTTS fallback)
    tts_engine = TTSEngine(model_path=None)
    logger.info("TTS engine ready")

    try:
        fingerspell_model = keras.models.load_model(
            os.path.join(WEIGHTS_DIR, "../fingerspell_models/alphabet_pose_mlp_24letters.h5"),
            compile=False
        )
        fingerspell_labels = np.load(
            os.path.join(WEIGHTS_DIR, "../fingerspell_models/alphabet_labels_24letters.npy"),
            allow_pickle=True
        )
        logger.info("✅ Fingerspelling model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load fingerspelling model: {e}")

    logger.info("=" * 50)
    logger.info("Server ready at http://localhost:8000")
    logger.info("   Health:      http://localhost:8000/health")
    logger.info("   Video:       http://localhost:8000/translate/video")
    logger.info("   Fingerspell: http://localhost:8000/translate/fingerspell")
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TranslationResponse(BaseModel):
    """Translation result with predicted sentence, tokens, confidence, and synthesized audio."""
    sentence: str
    confidence: float
    tokens: list[str]
    audio_base64: str


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def decode_tokens(token_ids: torch.Tensor) -> list[str]:
    """Convert a 1D tensor of token IDs to a list of token strings, stopping at EOS."""
    tokens = []
    for tid in token_ids.tolist():
        if tid == EOS_ID:
            break
        tok = ID_TO_TOKEN.get(tid, f"<UNK:{tid}>")
        # Skip SOS/PAD tokens in output
        if tok in ("<sos>", "<SOS>", "<pad>", "<PAD>"):
            continue
        tokens.append(tok)
    return tokens


def tokens_to_sentence(tokens: list[str]) -> str:
    """Look up the token tuple in TOKEN_SEQ_TO_PHRASE; fallback to joining with spaces."""
    key = tuple(tokens)
    if key in TOKEN_SEQ_TO_PHRASE:
        return TOKEN_SEQ_TO_PHRASE[key]
    # Fallback: join tokens with spaces
    return " ".join(tokens)


def run_inference(landmarks_np: np.ndarray) -> TranslationResponse:
    """Run the full inference pipeline: Landmarks -> Model -> Sentence -> TTS -> Response"""
    # Convert MediaPipe (T, 1662) to UniSign dict format
    src = mediapipe_to_unisign(landmarks_np)

    # Move tensors to device
    src = {k: v.to(DEVICE) for k, v in src.items()}

    # Run autoregressive inference
    token_ids, confidence = model.inference(src, SOS_ID, EOS_ID)

    # Decode tokens (batch size = 1)
    tokens = decode_tokens(token_ids[0])
    conf = confidence[0].item()

    if not tokens or conf < CONFIDENCE_THRESHOLD:
        return TranslationResponse(
            sentence="Unrecognized sign",
            confidence=round(conf, 4),
            tokens=tokens,
            audio_base64="",
        )

    sentence = tokens_to_sentence(tokens)

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
        tokens=tokens,
        audio_base64=audio_b64,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint — confirms the server is online."""
    return {
        "service": "Zora - Recorded Video Translation API",
        "status": "online",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "translate_video": "/translate/video",
            "translate_fingerspell": "/translate/fingerspell",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "vocab_size": len(VOCAB) if VOCAB else 0,
        "model": "MeetixSeq2Seq",
    }



@app.post("/translate/video", response_model=TranslationResponse)
async def translate_video(file: UploadFile = File(...)):
    """Upload a video file. The server extracts MediaPipe landmarks, then runs inference."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="opencv-python and mediapipe are required for video processing",
        )

    # Save uploaded file temporarily
    contents = await file.read()
    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
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

            # Face (468 landmarks * 3 = 1404 values)
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 1404)

            # Pose (33 landmarks * 4 = 132 values: x, y, z, visibility)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                frame_landmarks.extend([0.0] * 132)

            # Left hand (21 landmarks * 3 = 63 values)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)

            # Right hand (21 landmarks * 3 = 63 values)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)

            # Total: 1404 + 132 + 63 + 63 = 1662
            assert len(frame_landmarks) == 1662, f"Expected 1662, got {len(frame_landmarks)}"
            all_frames.append(frame_landmarks)

    cap.release()
    os.remove(tmp_path)

    if not all_frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    landmarks_np = np.array(all_frames, dtype=np.float32)
    return run_inference(landmarks_np)


@app.post("/translate/fingerspell", response_model=TranslationResponse)
async def translate_fingerspell(file: UploadFile = File(...)):
    """Upload a video file for fingerspelling translation."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="opencv-python and mediapipe are required for video processing",
        )

    # Save uploaded file temporarily
    contents = await file.read()
    tmp_path = os.path.join(tempfile.gettempdir(), f"fs_upload_{file.filename}")
    with open(tmp_path, "wb") as f:
        f.write(contents)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(tmp_path)
    
    letter_buffer = []
    raw_pred_buffer = deque(maxlen=3)
    last_detected_letter = None
    letter_hold_frames = 0
    CONFIDENCE_THRESHOLDS = {'N': 0.75, 'M': 0.72, 'T': 0.72, 'S': 0.70, 'default': 0.60}

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                lm_list = [[p.x, p.y, p.z] for p in lm.landmark]
                x_raw = np.array([c for pt in lm_list for c in pt], dtype=np.float32)
                
                try:
                    features = featurize_pose(x_raw).reshape(1, -1)
                    preds = fingerspell_model.predict(features, verbose=0)[0]
                    idx = np.argmax(preds)
                    conf = float(preds[idx])
                    detected_raw = fingerspell_labels[idx]
                    
                    req_conf = CONFIDENCE_THRESHOLDS.get(detected_raw, CONFIDENCE_THRESHOLDS['default'])
                    if conf < req_conf:
                        continue
                        
                    raw_pred_buffer.append(idx)
                    if len(raw_pred_buffer) >= 3:
                        idx_max = max(set(list(raw_pred_buffer)), key=list(raw_pred_buffer).count)
                        detected = fingerspell_labels[idx_max]
                        
                        if detected == last_detected_letter:
                            letter_hold_frames += 1
                        else:
                            last_detected_letter = detected
                            letter_hold_frames = 1
                            
                        threshold = 15
                        if conf > 0.85:
                            threshold = 8
                        if letter_buffer and detected == letter_buffer[-1]:
                            threshold = 45
                            
                        if letter_hold_frames == threshold:
                            letter_buffer.append(detected)
                            letter_hold_frames = 0
                except Exception as e:
                    logger.error(f"Fingerspell frame error: {e}")

    cap.release()
    os.remove(tmp_path)
    
    word = "".join(letter_buffer)
    if not word:
        word = "Unrecognized sign"
        
    audio_b64 = ""
    if word != "Unrecognized sign":
        try:
            audio_bytes = tts_engine.synthesize(word)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    return TranslationResponse(
        sentence=word,
        confidence=1.0,
        tokens=list(word),
        audio_base64=audio_b64,
    )



# ---------------------------------------------------------------------------

# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 50)
    print("  Zora - Sign Language Translation Server v2")
    print("  (Seq2Seq Architecture)")
    print("=" * 50)
    print()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
