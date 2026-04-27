from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from typing import List, Dict, Any, Tuple

import librosa
import numpy as np

app = FastAPI(title="ChordSense Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CHORD_FINGERINGS = {
    "C": "x32010", "Cm": "x35543", "D": "xx0232", "Dm": "xx0231",
    "E": "022100", "Em": "022000", "F": "133211", "Fm": "133111",
    "F#": "244322", "F#m": "244222", "G": "320003", "Gm": "355333",
    "A": "x02220", "Am": "x02210", "Bb": "x13331", "Bbm": "x13321",
    "B": "x24442", "Bm": "x24432",

    "C7": "x32310", "D7": "xx0212", "E7": "020100", "F7": "131211",
    "G7": "320001", "A7": "x02020", "B7": "x21202",

    "Cmaj7": "x32000", "Dmaj7": "xx0222", "Emaj7": "021100",
    "Fmaj7": "xx3210", "Gmaj7": "320002", "Amaj7": "x02120",
    "Bbmaj7": "x13231", "Bmaj7": "x24342",

    "Cm7": "x35343", "Dm7": "xx0211", "Em7": "022030",
    "Fm7": "131111", "Gm7": "353333", "Am7": "x02010",
    "Bbm7": "x13121", "Bm7": "x20202",
}

CHORD_PATTERNS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
}

COMMON_SUFFIX_WEIGHT = {
    "": 0.045,
    "m": 0.045,
    "7": 0.010,
    "maj7": -0.010,
    "m7": -0.006,
}

def build_templates() -> Dict[str, np.ndarray]:
    templates = {}
    for root_index, root in enumerate(NOTE_NAMES):
        for suffix, intervals in CHORD_PATTERNS.items():
            vec = np.zeros(12)
            for interval in intervals:
                vec[(root_index + interval) % 12] = 1.0
            templates[root + suffix] = vec
    return templates

ALL_TEMPLATES = build_templates()

def normalize(v: np.ndarray) -> np.ndarray:
    total = float(np.sum(v))
    return v / total if total > 0 else v

def split_chord(chord: str) -> Tuple[str, str]:
    if len(chord) >= 2 and chord[1] in ["#", "b"]:
        return chord[:2], chord[2:]
    return chord[:1], chord[1:]

def simplify_chord(chord: str) -> str:
    root, suffix = split_chord(chord)

    if suffix in ["maj7", "7"]:
        return root
    if suffix == "m7":
        return root + "m"

    return chord

def estimate_chord(chroma_vec: np.ndarray) -> str:
    chroma_vec = normalize(chroma_vec)

    scores = []

    for name, template in ALL_TEMPLATES.items():
        root, suffix = split_chord(name)
        template_norm = normalize(template)

        score = float(np.dot(chroma_vec, template_norm))
        score += COMMON_SUFFIX_WEIGHT.get(suffix, -0.02)

        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scores[0]
    simplified = simplify_chord(best_name)

    # If a simple version is close enough, prefer it.
    for name, score in scores[:8]:
        if name == simplified and best_score - score < 0.08:
            return name

    return simplified

def estimate_key(chroma_mean: np.ndarray) -> str:
    idx = int(np.argmax(chroma_mean))
    return f"{NOTE_NAMES[idx]} Major"

def format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"

def safe_tempo_value(raw: Any) -> int:
    if isinstance(raw, np.ndarray):
        raw = float(raw.flat[0]) if raw.size > 0 else 92
    try:
        tempo = int(round(float(raw)))
        return tempo if tempo > 0 else 92
    except Exception:
        return 92

def merge_consecutive_chords(chords: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not chords:
        return []

    merged = [chords[0]]
    for chord in chords[1:]:
        if chord["chord"] != merged[-1]["chord"]:
            merged.append(chord)

    return merged

def remove_fast_noise(chords: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(chords) < 3:
        return chords

    cleaned = []
    for i, chord in enumerate(chords):
        if 0 < i < len(chords) - 1:
            prev_chord = chords[i - 1]["chord"]
            next_chord = chords[i + 1]["chord"]
            current = chord["chord"]

            if prev_chord == next_chord and current != prev_chord:
                continue

        cleaned.append(chord)

    return cleaned

@app.get("/")
def root():
    return {"message": "ChordSense backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    difficulty: str = Form("beginner"),
    sourceType: str = Form("fileImport"),
):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        tmp.write(await audio.read())

    try:
        print(f"Analyzing file: {audio.filename}")

        y, sr = librosa.load(temp_path, sr=16000, mono=True)

        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Empty audio")

        duration = float(librosa.get_duration(y=y, sr=sr))
        print(f"Duration: {duration:.2f}s")

        raw_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = safe_tempo_value(raw_tempo)

        # Harmonic separation helps reduce drums/percussion.
        y_harmonic, _ = librosa.effects.hpss(y)

        # STFT chroma is lighter than CQT and works better on Render memory.
        chroma = librosa.feature.chroma_stft(
            y=y_harmonic,
            sr=sr,
            n_fft=4096,
            hop_length=2048,
        )

        if chroma is None or chroma.size == 0:
            raise HTTPException(status_code=400, detail="Could not extract chroma.")

        frame_count = chroma.shape[1]
        seconds_per_frame = 2048 / sr

        # Analyze roughly every 6 seconds.
        window_seconds = 6
        window_frames = max(1, int(window_seconds / seconds_per_frame))

        chords = []
        for start in range(0, frame_count, window_frames):
            end = min(start + window_frames, frame_count)
            section = chroma[:, start:end]

            if section.shape[1] == 0:
                continue

            section_vec = np.median(section, axis=1)
            chord_name = estimate_chord(section_vec)
            time_sec = start * seconds_per_frame

            chords.append({
                "chord": chord_name,
                "fingering": CHORD_FINGERINGS.get(chord_name, "x32010"),
                "time": format_time(time_sec),
            })

        chords = merge_consecutive_chords(chords)
        chords = remove_fast_noise(chords)
        chords = merge_consecutive_chords(chords)

        # Limit display to avoid clutter, but keep real progression.
        chords = chords[:32]

        key_signature = estimate_key(np.mean(chroma, axis=1))

        result = {
            "title": audio.filename or "Song",
            "keySignature": key_signature,
            "tempo": tempo,
            "difficulty": difficulty,
            "sourceType": sourceType,
            "chords": chords,
        }

        print("Analysis complete")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)