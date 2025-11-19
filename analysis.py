import pandas as pd
from pathlib import Path

def calculate_average_hands(df):
    """
    Calculates the average position of the left and right hands for each person in each frame.
    """

    left_hand_ids = [5, 7, 9]  # left_shoulder, left_elbow, left_wrist
    right_hand_ids = [6, 8, 10]  # right_shoulder, right_elbow, right_wrist

    # Filter for hand keypoints
    hand_keypoints = df[df['keypoint_id'].isin(left_hand_ids + right_hand_ids)]

    # Initialize list to store average hand data
    average_hands_data = []

    # Group by frame and person_id
    for (frame, person_id), group in hand_keypoints.groupby(['frame', 'person_id']):

        left_hand = group[group['keypoint_id'].isin(left_hand_ids)]
        right_hand = group[group['keypoint_id'].isin(right_hand_ids)]

        if not left_hand.empty:
            avg_x_left = left_hand['x'].mean()
            avg_y_left = left_hand['y'].mean()
            average_hands_data.append([frame, person_id, 'left_hand', avg_x_left, avg_y_left])

        if not right_hand.empty:
            avg_x_right = right_hand['x'].mean()
            avg_y_right = right_hand['y'].mean()
            average_hands_data.append([frame, person_id, 'right_hand', avg_x_right, avg_y_right])

    # Create a new DataFrame for the average hand positions
    avg_hands_df = pd.DataFrame(average_hands_data, columns=['frame', 'person_id', 'hand', 'avg_x', 'avg_y'])

    return avg_hands_df

# ----- Main execution -----
csv_path = Path("keypoints_data (2).csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Keypoints data file not found: {csv_path}")

df = pd.read_csv(csv_path)

# Calculate average hand positions
avg_hands_df = calculate_average_hands(df)

# Save the results to a new CSV file
output_csv_path = Path("average_hands.csv")
avg_hands_df.to_csv(output_csv_path, index=False)

print(f"Average hand positions saved to: {output_csv_path}")
