import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ----- konfiguracja -----
csv_path = Path("keypoints_data (2).csv")
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

# Lista nazw punktów (tak jak zapisywałeś)
keypoint_names = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle"
]

# Połączenia szkieletu (indeksy w keypoint_id)
skeleton_pairs = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Połączenia dla rakiety (góra i dół)
racket_pairs = [
    (9, 10),  # Połączenie między nadgarstkami
]

# ----- wczytanie CSV -----
df = pd.read_csv(csv_path)

required_cols = {"frame","person_id","keypoint_id","x","y"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Brakuje kolumn: {required_cols - set(df.columns)}")

# Ustal stałą liczbę keypointów (len(keypoint_names))
N_KP = len(keypoint_names)  # 17

# Lista wszystkich person w całym nagraniu (np. [0,1])
persons = sorted(df["person_id"].unique().tolist())
n_persons = len(persons)
print(f"Znaleziono osoby: {persons}")

# Kolory (rozszerzalne)
base_colors = ["red","blue","green","orange","purple","pink","brown","gray"]
if n_persons > len(base_colors):
    colors = (base_colors * ((n_persons//len(base_colors))+1))[:n_persons]
else:
    colors = base_colors[:n_persons]

# ----- przygotuj listę klatek (posortowane) -----
frame_ids = sorted(df["frame"].unique().tolist())
print(f"Liczba klatek do animacji: {len(frame_ids)}")

# ----- przygotuj początkowe trace'y (stała liczba: 2 trace'y na osobę -> markers + skeleton lines) -----
initial_traces = []
for idx, pid in enumerate(persons):
    # # markers trace (pokażemy legendę tylko raz na początku)
    # initial_traces.append(go.Scatter(
    #     x=[], y=[],
    #     mode="markers+text",
    #     text=[], textposition="top center",
    #     marker=dict(size=8),
    #     name=f"Person {pid}",
    #     marker_symbol="circle",
    #     marker_line_width=1,
    #     marker_line_color="black",
    #     hoverinfo="text"
    # ))
    # # skeleton trace (linie)
    # initial_traces.append(go.Scatter(
    #     x=[], y=[],
    #     mode="lines",
    #     line=dict(width=2),
    #     name=f"skeleton {pid}",
    #     showlegend=False  # nie duplikuj legendy
    # ))
    # racket trace (linie)
    initial_traces.append(go.Scatter(
        x=[], y=[],
        mode="lines",
        line=dict(width=4, color="purple"),
        name=f"racket {pid}",
        showlegend=False
    ))

# ----- zbuduj frames (z danymi w tej samej kolejności co initial_traces) -----
frames = []
for f in frame_ids:
    df_f = df[df["frame"] == f]
    frame_data = []
    for pid in persons:
        df_p = df_f[df_f["person_id"] == pid]
        # Fill full array of shape (N_KP, 2) with NaN, wypełnij znanymi kp
        pts = np.full((N_KP, 2), np.nan)
        texts = [""] * N_KP
        if not df_p.empty:
            # Upewnijmy się, że keypoint_id są ints
            for _, row in df_p.iterrows():
                kid = int(row["keypoint_id"])
                if 0 <= kid < N_KP:
                    pts[kid, 0] = float(row["x"])
                    pts[kid, 1] = float(row["y"])
                    # opcjonalny tekst: "id:kp"
                    texts[kid] = f"{pid}:{kid}"
        # # markers trace data
        # frame_data.append(go.Scatter(x=pts[:,0].tolist(),
        #                              y=pts[:,1].tolist(),
        #                              text=texts,
        #                              mode="markers+text",
        #                              textposition="top center",
        #                              marker=dict(size=8, color=colors[persons.index(pid)]),
        #                              name=f"Person {pid}",
        #                              hoverinfo="text"))
        # # build skeleton lines with None separators
        # x_lines, y_lines = [], []
        # for a,b in skeleton_pairs:
        #     if a < N_KP and b < N_KP:
        #         xa, ya = pts[a,0], pts[a,1]
        #         xb, yb = pts[b,0], pts[b,1]
        #         # tylko narysuj segment jeśli obie współrzędne nie są NaN
        #         if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
        #             x_lines += [float(xa), float(xb), None]
        #             y_lines += [float(ya), float(yb), None]
        # frame_data.append(go.Scatter(x=x_lines, y=y_lines, mode="lines",
        #                              line=dict(width=2, color=colors[persons.index(pid)]),
        #                              showlegend=False))
        # build racket lines
        x_racket, y_racket = [], []
        for a, b in racket_pairs:
            if a < N_KP and b < N_KP:
                xa, ya = pts[a, 0], pts[a, 1]
                xb, yb = pts[b, 0], pts[b, 1]
                if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
                    x_racket += [float(xa), float(xb), None]
                    y_racket += [float(ya), float(yb), None]
        frame_data.append(go.Scatter(x=x_racket, y=y_racket, mode="lines",
                                     line=dict(width=4, color="purple"),
                                     showlegend=False))
    frames.append(go.Frame(data=frame_data, name=str(f)))

# ----- ustawienia osi -----
# Dobierz rozstaw na podstawie całego zbioru (dodaj margines)
x_min, x_max = float(df["x"].min()), float(df["x"].max())
y_min, y_max = float(df["y"].min()), float(df["y"].max())
x_margin = (x_max - x_min) * 0.05 if x_max>x_min else 10
y_margin = (y_max - y_min) * 0.05 if y_max>y_min else 10

x_range = [max(0, x_min - x_margin), x_max + x_margin]
y_range = [y_max + y_margin, max(0, y_min - y_margin)]  # odwrócona oś Y (top = 0 visually)

# ----- stwórz figurę -----
fig = go.Figure(
    data=initial_traces,
    frames=frames,
    layout=go.Layout(
        title="Dynamiczna wizualizacja rakiet (Plotly)",
        xaxis=dict(range=x_range, title="x"),
        yaxis=dict(range=y_range, title="y", autorange=False),
        width=900,
        height=600,
        updatemenus=[{
            "buttons": [
                {"args":[None, {"frame":{"duration":50, "redraw":True}, "fromcurrent":True}],
                 "label":"Start", "method":"animate"},
                {"args":[[None], {"frame":{"duration":0, "redraw":False}, "mode":"immediate"}],
                 "label":"Stop", "method":"animate"}
            ],
            "direction":"left",
            "pad":{"r":10,"t":70},
            "showactive":False,
            "type":"buttons",
            "x":0.1, "xanchor":"right", "y":0, "yanchor":"top"
        }]
    )
)

# ustawienia osi odwróconej (jeśli obrazy mają org Y w górę)
fig.update_yaxes(autorange=False)

# Opcjonalnie: dodaj pierwszy frame jako widoczny stan początkowy
if frames:
    fig.update_traces(selector=dict(mode="markers+text"), overwrite=True)  # utrzymaj styl
    fig.frames = frames

# Wyświetl i zapisz
fig.show()
fig.write_html("racket_only_visualization.html", include_plotlyjs="cdn")
print("Gotowe — zapisano racket_only_visualization.html")
