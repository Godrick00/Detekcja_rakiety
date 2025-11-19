import pandas as pd
from pathlib import Path
import numpy as np

def detect_handedness(avg_hands_df):
    """
    Detects the handedness of each player based on the total distance traveled by each hand.
    """

    handedness_results = {}

    # Group by person_id
    for person_id, group in avg_hands_df.groupby('person_id'):

        left_hand_df = group[group['hand'] == 'left_hand'].sort_values('frame')
        right_hand_df = group[group['hand'] == 'right_hand'].sort_values('frame')

        # Calculate total distance for left hand
        left_hand_dist = 0
        if not left_hand_df.empty:
            x_left = left_hand_df['avg_x'].to_numpy()
            y_left = left_hand_df['avg_y'].to_numpy()
            left_hand_dist = np.sum(np.sqrt(np.diff(x_left)**2 + np.diff(y_left)**2))

        # Calculate total distance for right hand
        right_hand_dist = 0
        if not right_hand_df.empty:
            x_right = right_hand_df['avg_x'].to_numpy()
            y_right = right_hand_df['avg_y'].to_numpy()
            right_hand_dist = np.sum(np.sqrt(np.diff(x_right)**2 + np.diff(y_right)**2))

        # Determine handedness
        if right_hand_dist > left_hand_dist:
            handedness_results[person_id] = 'right-handed'
        elif left_hand_dist > right_hand_dist:
            handedness_results[person_id] = 'left-handed'
        else:
            handedness_results[person_id] = 'undetermined'

    return handedness_results

# ----- Main execution -----
average_hands_csv_path = Path("average_hands.csv")
if not average_hands_csv_path.exists():
    raise FileNotFoundError(f"Average hands data file not found: {average_hands_csv_path}")

avg_hands_df = pd.read_csv(average_hands_csv_path)

# Detect handedness
handedness = detect_handedness(avg_hands_df)

# Print results
for person_id, hand in handedness.items():
    print(f"Person {person_id} is likely {hand}.")
