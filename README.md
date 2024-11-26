# Crossy Road Course Project [CV-component]

## Overview
This project is a computer vision-based implementation for detecting game-over screens in the Crossy Road game. It includes functionality for mapping detected objects to a grid and visualizing the results.

## Features
• Fine-tuned Object Detection: Leverages YOLOv11 for precise object detection within game screenshots, enabling real-time identification of game elements.<br>
• Game Over Detection: Implements a template-matching approach to reliably detect the game-over button. This method focuses on a fixed region of interest, optimizing detection speed and accuracy.<br>
• Grid Mapping: Transforms detected objects into a grid representation by calculating their bounding box coordinates, providing a structured input format for further analysis.<br>


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


