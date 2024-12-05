# Crossy Road RL Agent  

## Project Overview  
This project aims to develop an AI agent that plays the mobile game **Crossy Road** using Reinforcement Learning (RL) and Computer Vision (CV) techniques. The agent's main objective is to navigate the game environment, avoiding obstacles and successfully crossing the road.  

## Repository  
The source code and related artifacts can be found at: [GitHub Repository](https://github.com/1kkiRen/PMLDL-Course-Project)  

## Artifacts  
- **[Repository](https://github.com/1kkiRen/Crossy-Road-Course-Project)**: Contains the main codebase for the project.  
- **[Dataset](https://github.com/1kkiRen/Crossy-Road-Course-Project/tree/cv-main/datasets/For_finetuning_yolov11)**: Custom dataset of annotated game screenshots.  
- **[Script for Screenshots Collection](https://github.com/1kkiRen/Crossy-Road-Course-Project/blob/main/screenshots_collector.py)**: Python script for capturing game screenshots at regular intervals.  
- **[CV_Models](https://github.com/1kkiRen/Crossy-Road-Course-Project/tree/cv-main/src/models)**: Models developed for object detection.  
- **[RL_Models](https://github.com/1kkiRen/Crossy-Road-Course-Project/tree/rl-main)**: Models developed for reinforcement learning.  

## Problem Statement and Objective  
In Crossy Road, the player controls a hen that must cross a busy road while avoiding obstacles. Our goal was to create a reinforcement learning agent capable of mastering this game. We experimented with various techniques, starting with raw screenshots and evolving to a grid-based representation of the game environment to enhance the agent's understanding.  

## Methodology  
The project consists of two main components:  

### 1. Reinforcement Learning (RL) Component  
- **Environment**: A continuous scrolling road with various obstacles.  
- **State-space**: Includes the game screenshot, the position of the hen, and the position of obstacles.  
- **Action-space**: The agent can move left, right, backward, or forward.  
- **Reward Function Design**:  
  - Positive Rewards: Encourages forward movement and strategic patience.  
  - Negative Rewards: Penalizes game over and inefficient movements.  

### 2. Computer Vision (CV) Component  
- **Object Detection with visual model**: Identifies key elements in the game, including the hen and obstacles.
- **Template Matching Method**: Used for restart button detection.

[Watch the Crossy Road obstacles Detection ](https://drive.google.com/file/d/1edZ-lxuQySawbTdM7cc1RcK6ALIzMCEx/preview)

## Iterations and Improvements  
1. **1st Iteration**: Direct Screenshot Input  
   - Input: Raw game screenshots.  
   - Outcome: Ineffective; the model failed to converge.  

2. **2nd Iteration**: Detected Objects  
   - Input: Detected objects from CV pipeline.  
   - Outcome: Some improvement, but performance limited.  

3. **3rd Iteration**: Grid Representation  
   - Input: Transformed game field into a grid.  
   - Outcome: Effective; improved training speed and model performance.  

4. **4th Iteration**: R-IQN with Enhanced Environment  
   - Input: Replaced DQN with R-IQN and increased class labels.  
   - Outcome: Highly effective; significant improvement in adaptability and performance.  

## Results  
The fourth iteration achieved the best performance, with the agent scoring **29 points** in the Crossy Road game. The use of the R-IQN model allowed for better handling of sequential dependencies and improved decision-making.  

[Watch the Crossy Road RL Agent Video](https://drive.google.com/file/d/1u5NwORIDgiYjUVQ0rwhrERhyxmoAB7RG/preview)

The detatiled report can be found [here]()


## Limitations  
- Object detection inaccuracies may affect the agent's performance.  
- The grid representation might oversimplify complex game dynamics.  
- The RL training process was computationally intensive, limited by available resources.  

## References  
For further reading and inspiration, explore the following related projects:  
- [Deep Learning Crossy Road](https://github.com/YilongSong/Deep-Learning-Crossy-Road) by Yilong Song  
- [CS221 Poster: Reinforcement Learning for Crossy Road](https://cs221.stanford.edu/posters/2021/Weerawardena_Tan_Rubin.pdf) by Sajana Weerawardena, Alwyn Tan, and Nick Rubin  
- [Deep Q-Learning Crossy Road](https://github.com/MarlonFacey/Deep-Q-Learning-Crossy-Road) by Marlon Facey et al.  

## License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.