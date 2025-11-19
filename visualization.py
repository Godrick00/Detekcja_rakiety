import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ----- Configuration -----
keypoints_csv_path = Path("yolo_results/pose_predict/keypoints_data.csv")
average_hands_csv_path = Path("average_hands.csv")

# ----- Load Data -----
if not keypoints_csv_path.exists():
    raise FileNotFoundError(f"Keypoints data file not found: {keypoints_csv_path}")
if not average_hands_csv_path.exists():
    raise FileNotFoundError(f"Average hands data file not found: {average_hands_csv_path}")

keypoints_df = pd.read_csv(keypoints_csv_path)
avg_hands_df = pd.read_csv(average_hands_csv_path)

# --- Create Visualization ---
fig = go.Figure()

# Add traces for each person's keypoints
persons = sorted(keypoints_df["person_id"].unique())
colors = ["red", "blue", "green", "orange", "purple", "pink", "brown", "gray"]
for person_id in persons:
    person_df = keypoints_df[keypoints_df['person_id'] == person_id]
    fig.add_trace(go.Scatter(
        x=person_df['x'],
        y=person_df['y'],
        mode='markers',
        name=f'Person {person_id} Keypoints',
        marker=dict(color=colors[person_id % len(colors)], size=5),
        visible=False  # Initially hidden
    ))

# Add traces for average hand positions
for person_id in persons:
    for hand in ['left_hand', 'right_hand']:
        hand_df = avg_hands_df[(avg_hands_df['person_id'] == person_id) & (avg_hands_df['hand'] == hand)]
        fig.add_trace(go.Scatter(
            x=hand_df['avg_x'],
            y=hand_df['avg_y'],
            mode='markers',
            name=f'Person {person_id} Avg {hand}',
            marker=dict(color=colors[person_id % len(colors)], symbol='cross', size=10),
            visible=False  # Initially hidden
        ))

# Add trace for racket
racket_df = keypoints_df[keypoints_df['keypoint_name'] == 'tennis racket']
fig.add_trace(go.Scatter(
    x=racket_df['x'],
    y=racket_df['y'],
    mode='markers',
    name='Racket',
    marker=dict(color='black', symbol='square', size=8),
    visible=False  # Initially hidden
))

# Make the first trace visible
if len(fig.data) > 0:
    fig.data[0].visible = True

# Create dropdown menu
updatemenus = [
    {
        'buttons': [
            {
                'method': 'update',
                'label': 'All Keypoints',
                'args': [{'visible': [True if 'Keypoints' in trace.name else False for trace in fig.data]}]
            },
            {
                'method': 'update',
                'label': 'Average Hands',
                'args': [{'visible': [True if 'Avg' in trace.name else False for trace in fig.data]}]
            },
            {
                'method': 'update',
                'label': 'Racket',
                'args': [{'visible': [True if 'Racket' in trace.name else False for trace in fig.data]}]
            }
        ],
        'direction': 'down',
        'showactive': True,
    }
]

fig.update_layout(
    updatemenus=updatemenus,
    title="Player Movement Visualization",
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    yaxis=dict(autorange="reversed") # Invert Y axis for intuitive display
)


# Save to HTML
fig.write_html("interactive_visualization.html")
print("Interactive visualization saved to interactive_visualization.html")
