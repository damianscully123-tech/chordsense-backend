from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from typing import List, Dict, Any

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

CHORD_FINGERINGS = {
    "C": "x32010",
    "Cm": "x35543",
    "C7": "x32310",
    "Cmaj7": "x32000",
    "Cm7": "x35343",

    "C#": "x46664",
    "C#m": "x46654",
    "C#7": "x46464",
    "C#maj7": "x46564",
    "C#m7": "x46454",

    "D": "xx0232",
    "Dm": "xx0231",
    "D7": "xx0212",
    "Dmaj7": "xx0222",
    "Dm7": "xx0211",

    "Eb": "x65343",
    "Ebm": "x68876",
    "Eb7": "x65646",
    "Ebmaj7": "x65756",
    "Ebm7": "x68676",

    "E": "022100",
    "Em": "022000",
    "E7": "020100",
    "Emaj7": "021100",
    "Em7": "022030",

    "F": "133211",
    "Fm": "133111",
    "F7": "131211",
    "Fmaj7": "xx3210",
    "Fm7": "131111",

    "F#": "244322",
    "F#m": "244222",
    "F#7": "242322",
    "F#maj7": "243322",
    "F#m7": "242222",

    "G": "320003",
    "Gm": "355333",
    "G7": "320001",
    "Gmaj7": "320002",
    "Gm7": "353333",

    "Ab": "466544",
    "Abm": "466444",
    "Ab7": "464544",
    "Abmaj7": "465544",
    "Abm7": "464444",

    "A": "x02220",
    "Am": "x02210",
    "A7": "x02020",
    "Amaj7": "x02120",
    "Am7": "x02010",

    "Bb": "x13331",
    "Bbm": "x13321",
    "Bb7": "x13131",
    "Bbmaj7": "x13231",
    "Bbm7": "x13121",

    "B": "x24442",
    "Bm": "x24432",
    "B7": "x21202",
    "Bmaj7": "x24342",
    "Bm7": "x20202",
}

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CHORD_PATTERNS = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
}


def build_chord_templates():
    templates = {}

    for root_index, root_name in enumerate(NOTE_NAMES):
        for suffix, intervals in CHORD_PATTERNS.items():
            vec = np.zeros(12, dtype=float)
            for interval in intervals:
                vec[(root_index + interval) % 12] = 1.0
            templates[f"{root_name}{suffix}"] = vec

    return templates


ALL_TEMPLATES = build_chord_templates()


def normalize(v):
    s = np.sum(v)
    return v / s if s > 0 else v


def estimate_key(chroma_mean):
    idx = int(np.argmax(chroma_mean))
    return f"{NOTE_NAMES[idx]} Major"


def estimate_chord(chroma_vec):
    chroma_vec = normalize(chroma_vec)

    best_name = "C"
    best_score = -1.0

    for name, template in ALL_TEMPLATES.items():
        score = float(np.dot(chroma_vec, normalize(template)))

        # ✅ SMALL bonus for common musical chords ONLY
        if any(tag in name for tag in ["7", "maj7", "m7"]):
            score += 0.005

        if score > best_score:
            best_score = score
            best_name = name

    return best_name


def segment_times(duration_sec, count):
    if count <= 0:
        return []

    step = duration_sec / count
    out = []

    for i in range(count):
        total = int(i * step)
        minutes = total // 60
        seconds = total % 60
        out.append(f"{minutes}:{seconds:02d}")

    return out


def safe_tempo_value(raw):
    if isinstance(raw, np.ndarray):
        raw = float(raw.flat[0]) if raw.size > 0 else 92

    try:
        tempo = int(round(float(raw)))
        return tempo if tempo > 0 else 92
    except:
        return 92


def merge_consecutive_chords(chords):
    if not chords:
        return []

    merged = [chords[0]]

    for c in chords[1:]:
        if c["chord"] != merged[-1]["chord"]:
            merged.append(c)

    return merged


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
        y, sr = librosa.load(temp_path, sr=22050, mono=True)

        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Empty audio")

        duration = librosa.get_duration(y=y, sr=sr)

        raw_tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = safe_tempo_value(raw_tempo)

        # 🔥 BETTER AUDIO ANALYSIS
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        if len(beat_frames) < 2:
            beat_frames = np.arange(0, chroma.shape[1], max(1, chroma.shape[1] // 16))

        beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

        max_sections = 16
        section_count = min(max_sections, beat_chroma.shape[1])

        bounds = np.linspace(0, beat_chroma.shape[1], num=section_count + 1, dtype=int)

        chords = []
        times = segment_times(duration, section_count)

        for i in range(section_count):
            start, end = bounds[i], bounds[i + 1]
            section = beat_chroma[:, start:end]

            vec = (
                np.mean(section, axis=1)
                if section.shape[1] > 0
                else beat_chroma[:, min(start, beat_chroma.shape[1] - 1)]
            )

            chord_name = estimate_chord(vec)

            chords.append({
                "chord": chord_name,
                "fingering": CHORD_FINGERINGS.get(chord_name, "x32010"),
                "time": times[i],
            })

        chords = merge_consecutive_chords(chords)
        key = estimate_key(np.mean(chroma, axis=1))

        return {
            "title": audio.filename or "Song",
            "keySignature": key,
            "tempo": tempo,
            "difficulty": difficulty,
            "sourceType": sourceType,
            "chords": chords,
        }

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)