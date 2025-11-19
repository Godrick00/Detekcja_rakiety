from moviepy.video.io.VideoFileClip import VideoFileClip
from ultralytics import YOLO
import csv
import pandas as pd
from pathlib import Path


# -----------------------------
# Step 1: Wczytaj plik video
# -----------------------------
video_path = Path("Crouin_v_Cardenas__German_Open_2025_QFs__FREE_Full_Match_trimmed (2).mp4").resolve()

if not video_path.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku wideo: {video_path}")
print(f"Wczytano plik: {video_path}")

# -----------------------------
# Step 2: Przytnij fragment
# -----------------------------
trimmed_video_path = video_path.stem + "_trimmed.mp4"

trimmed_video_path = video_path

# -----------------------------
# Step 3: Załaduj model YOLO z keypointami
# -----------------------------
print("Wczytywanie modelu YOLOv8 Pose...")
model = YOLO("yolov8l-pose.pt")  # lub większy: yolov8m-pose.pt, yolov8l-pose.pt

# -----------------------------
# Step 4: Katalog wynikowy
# -----------------------------
output_base_dir = Path("yolo_results")
project_name = "pose_predict"
output_dir = output_base_dir / project_name
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Step 5: Wykryj osoby i keypointy
# -----------------------------
print("Uruchamianie detekcji keypointów...")
results = model.track(
    source=str(trimmed_video_path),
    save=False,
    project=str(output_base_dir),
    name=project_name,
    verbose=True,
    classes=[0, 38],  # detekcja osób osoby
    conf = 0.5 # ustawienie progu pewności dla detekcji osób
)

# -----------------------------
# Step 6: Zapisz keypointy do CSV z nazwami
# -----------------------------
csv_path = output_dir / "keypoints_data.csv"
print(f"Zapisuję keypointy do: {csv_path}")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "person_id", "keypoint_id", "keypoint_name", "x", "y", "conf"])

    for i, frame_result in enumerate(results):
            if frame_result.boxes is not None and frame_result.keypoints is not None:
                # Sprawdź, czy są dostępne ID śledzenia
                if frame_result.boxes.id is not None:
                    track_ids = frame_result.boxes.id.cpu().numpy().astype(int)
                    pts = frame_result.keypoints.xy.cpu().numpy()
                    confs = frame_result.keypoints.conf.cpu().numpy()

                    for pid_idx, track_id in enumerate(track_ids):
                        p_xy = pts[pid_idx]
                        p_conf = confs[pid_idx]
                        for kid, ((x, y), c) in enumerate(zip(p_xy, p_conf)):
                            name = keypoint_names[kid] if kid < len(keypoint_names) else f"kp_{kid}"
                            # Zapisujemy track_id zamiast tymczasowego person_id
                            writer.writerow([i, track_id, kid, name, x, y, c])

print(f"Zapisano dane keypointów z nazwami ({csv_path})")


# -----------------------------
# Step 10: Interaktywna wizualizacja keypointów w Plotly
# -----------------------------
import plotly.graph_objects as go

print("Tworzenie interaktywnej wizualizacji keypointów (Plotly)...")

# Przygotuj dane z detekcji
frames_data = []
for frame_result in results:
    if frame_result.keypoints is not None and len(frame_result.keypoints) > 0:
        pts = frame_result.keypoints.xy.cpu().numpy()[0]  # pierwsza osoba
        frames_data.append(pts)

if not frames_data:
    print("Brak danych keypointów do wizualizacji.")
else:
    # Szkielet połączeń (COCO)
    skeleton_pairs = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # ręce
        (5, 6), (11, 12), (5, 11), (6, 12),  # tułów
        (11, 13), (13, 15), (12, 14), (14, 16)  # nogi
    ]

    # Przygotuj ramki do animacji
    frames = []
    for i, pts in enumerate(frames_data):
        # połączenia
        x_lines, y_lines = [], []
        for a, b in skeleton_pairs:
            x_lines += [pts[a, 0], pts[b, 0], None]
            y_lines += [pts[a, 1], pts[b, 1], None]

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=pts[:, 0], y=pts[:, 1],
                    mode="markers+text",
                    text=[str(k) for k in range(len(pts))],
                    textposition="top center",
                    marker=dict(size=8, color="red")
                ),
                go.Scatter(
                    x=x_lines, y=y_lines,
                    mode="lines",
                    line=dict(color="gray", width=2)
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    # Ustawienia osi i layoutu
    height = results[0].orig_shape[0]
    width = results[0].orig_shape[1]

    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode="markers"),
            go.Scatter(x=[], y=[], mode="lines")
        ],
        layout=go.Layout(
            title="Dynamiczna wizualizacja keypointów (YOLOv8 Pose)",
            xaxis=dict(range=[0, width]),
            yaxis=dict(range=[height, 0]),  # odwrócona oś Y
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}],
                     "label": "Start", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                     "label": "Stop", "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        ),
        frames=frames
    )

    # Wyświetl interaktywny wykres w przeglądarce
    fig.show()
    print("Interaktywna wizualizacja otwarta w przeglądarce!")

# -----------------------------
# Interaktywna wizualizacja keypointów wszystkich osób z CSV
# -----------------------------

# Ścieżka do CSV z zapisanymi keypointami
csv_path = Path("yolo_results/pose_predict/keypoints_data.csv")
df = pd.read_csv(csv_path)

# Upewnij się, że wymagane kolumny istnieją
required_cols = {"frame", "person_id", "keypoint_id", "x", "y"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Brakuje wymaganych kolumn: {required_cols - set(df.columns)}")

# Definicja połączeń szkieletu (COCO)
skeleton_pairs = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Przygotuj kolory dla osób
persons = sorted(df["person_id"].unique())
colors = ["red", "blue", "green", "orange", "purple", "pink", "brown", "gray"]
while len(colors) < len(persons):
    colors += colors  # w razie więcej osób niż kolorów

# Przygotowanie ramek animacji
frames_data = []
frames = sorted(df["frame"].unique())
for f in frames:
    frame_data = []
    df_frame = df[df["frame"] == f]

    # Dla każdej osoby w tej klatce
    for pid in df_frame["person_id"].unique():
        df_p = df_frame[df_frame["person_id"] == pid]
        pts = df_p.sort_values("keypoint_id")[["x", "y"]].to_numpy()

        # Dodaj punkty
        frame_data.append(go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="markers+text",
            text=[f"{pid}:{i}" for i in range(len(pts))],
            textposition="top center",
            marker=dict(size=8, color=colors[pid]),
            name=f"Person {pid}"
        ))

        # Dodaj linie szkieletu
        x_lines, y_lines = [], []
        for a, b in skeleton_pairs:
            if a < len(pts) and b < len(pts):
                x_lines += [pts[a, 0], pts[b, 0], None]
                y_lines += [pts[a, 1], pts[b, 1], None]

        frame_data.append(go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color=colors[pid], width=2),
            showlegend=False
        ))

    frames_data.append(go.Frame(data=frame_data, name=str(f)))

# Ustal wymiary osi
width = int(df["x"].max() * 1.1)
height = int(df["y"].max() * 1.1)

# Utwórz figurę Plotly
fig = go.Figure(
    data=[],
    layout=go.Layout(
        title="Dynamiczna wizualizacja wszystkich osób (Plotly)",
        xaxis=dict(range=[0, width]),
        yaxis=dict(range=[height, 0]),  # odwrócona oś Y
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                 "label": "Start", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                     "label": "Stop", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    ),
    frames=frames_data
)

# Wyświetl i zapisz do pliku HTML
fig.show()
fig.write_html("multi_person_pose.html", include_plotlyjs="cdn", auto_play=False)
print("Animacja zapisana do multi_person_pose.html i otwarta w przeglądarce")
