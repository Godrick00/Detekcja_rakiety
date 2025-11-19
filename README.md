# Squash Player Analysis

This project analyzes a video of a squash match to extract player movement data, determine player handedness, and visualize the results.

## Overview

The project consists of the following components:

- **Pose and Racket Detection:** Uses a pre-trained YOLOv8 model to detect and track players and rackets in the video. The keypoint data is saved to a CSV file.
- **Data Analysis:** Calculates the average hand positions for each player to simplify movement analysis.
- **Handedness Detection:** Determines whether each player is right-handed or left-handed based on the total distance their hands travel.
- **Interactive Visualization:** Provides an interactive Plotly visualization to explore player movements, including individual keypoints, averaged hand positions, and the racket trajectory.

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Analysis Pipeline:**
   ```bash
   python racket_detection.py
   python analysis.py
   python handedness_detection.py
   python visualization.py
   ```
3. **View the Visualization:**
   Open the generated `interactive_visualization.html` file in your web browser to explore the results.

## Methodologies

### Handedness Detection

Handedness is determined by calculating the total distance traveled by each hand. The hand that moves a greater distance is considered the dominant hand. This is a simple but effective heuristic for determining handedness in sports like squash.

### Racket Detection

Racket detection is performed using a pre-trained YOLOv8 model that has been trained on the COCO dataset, which includes a "tennis racket" class. This allows for accurate detection without the need for custom model training.
