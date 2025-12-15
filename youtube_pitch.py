#youtube영상에서 해당 피치 매칭 구간 추출


import librosa
import numpy as np

# =========================
# 설정
# =========================
AUDIO_PATH = "./아리아나 두숟갈 천서진 세숟갈 #승헌쓰 #도레미챌린지.mp3"
SR = 22050

MIN_DURATION = 0.3  # 초
MAX_DURATION = 1.0   # 초
CENT_THRESHOLD = 50  # 반음의 절반

TARGET_NOTES = {
    "C4": 60,  "C4s": 61,
    "D4": 62,  "D4s": 63,
    "E4": 64,
    "F4": 65,  "F4s": 66,
    "G4": 67,  "G4s": 68,
    "A4": 69,  "A4s": 70,
    "B4": 71,
    "C5": 72,  "C5s": 73
}

# =========================
# 보조 함수
# =========================
def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)

def cent_diff(m1, m2):
    return abs((m1 - m2) * 100)

# =========================
# 오디오 로드
# =========================
y, sr = librosa.load(AUDIO_PATH, sr=SR)

# =========================
# Pitch estimation
# =========================
f0, voiced_flag, _ = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C"),
    fmax=librosa.note_to_hz("D6")
)

times = librosa.times_like(f0, sr=sr)

# =========================
# 프레임별 note 매핑
# =========================
frames = []

for t, f, v in zip(times, f0, voiced_flag):
    if not v or f is None:
        frames.append((t, None, None))
        continue

    midi = hz_to_midi(f)

    matched_note = None
    for note, target_midi in TARGET_NOTES.items():
        if cent_diff(midi, target_midi) <= CENT_THRESHOLD:
            matched_note = note
            break

    frames.append((t, f, matched_note))

# =========================
# 연속 구간 추출
# =========================
segments = []

current_note = None
start_time = None
freqs = []

for t, f, note in frames:
    if note == current_note and note is not None:
        freqs.append(f)
        continue

    # 구간 종료
    if current_note is not None:
        end_time = t
        duration = end_time - start_time

        if MIN_DURATION <= duration <= MAX_DURATION:
            segments.append({
                "note": current_note,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "duration": round(duration, 3),
                "avg_freq": round(float(np.mean(freqs)), 2)
            })

    # 새 구간 시작
    if note is not None:
        current_note = note
        start_time = t
        freqs = [f]
    else:
        current_note = None
        start_time = None
        freqs = []

# =========================
# 출력
# =========================
print("🎯 C4 ~ C#5 (반음 포함), 0.3~1.0초 지속 구간")
print("-" * 70)

for s in segments:
    print(
        f"음: {s['note']:>3} | "
        f"시작: {s['start']:>6}s | "
        f"끝: {s['end']:>6}s | "
        f"길이: {s['duration']:>4}s | "
        f"평균 pitch: {s['avg_freq']} Hz"
    )

print(f"\n총 검출 구간 수: {len(segments)}")
