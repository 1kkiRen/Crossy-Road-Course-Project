# Crossy Road Course Project [CV-component]

## Overview
This project is a computer vision-based implementation for detecting game-over screens in the Crossy Road game. It includes functionality for mapping detected objects to a grid and visualizing the results.

## Features
• **Fine-tuned Object Detection**: Leverages YOLOv11 for precise object detection within game screenshots, enabling real-time identification of game elements.<br>
• **Game Over Detection**: Implements a template-matching approach to reliably detect the game-over button. This method focuses on a fixed region of interest, optimizing detection speed and accuracy.<br>
• **Grid Mapping**: Transforms detected objects into a grid representation by calculating their bounding box coordinates, providing a structured input format for further analysis.<br>

## Project Structure

- **src**: Contains the main source code files.
  - `end_of_game_detection.py`: Script for detecting game-over screens.
  - `detect_objects_with_api.py`: Script for detecting objects using the Roboflow inference API.

- **models**: Contains YOLO models and Jupyter Notebooks with experiments.


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/1kkiRen/Crossy-Road-Course-Project.git
   git checkout -b cv-component
   ```

2. Setup the environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use `venv\Scripts\activate`
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Game Over Detection
Run the `end_of_game_detection.py` script to detect game-over screens.

### YOLO Models
Explore the `models/...` files to use YOLO models for object detection.


