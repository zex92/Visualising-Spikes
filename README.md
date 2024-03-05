# Project Overview: Brain-Machine Interface (BMI) Spring 2024 - Neural Decoding Competition

## Introduction
Welcome to the BMI Spring 2024 Neural Decoding Competition. In this project, we aim to develop a machine learning model capable of predicting the trajectory of a monkey's hand based on neural activity data. This endeavor is part of a broader exploration into brain-machine interfacing and offers a practical and challenging task in neural decoding.

### Competition Description
Participants will design a neural decoder to drive a hypothetical prosthetic device by analyzing spike train data recorded from a monkey's brain. The goal is to precisely estimate the X & Y position of the monkey's hand as it reaches for targets during an arm movement task. The dataset comprises spike trains from 98 neural units and positional data of the monkey's hand over 632 time steps.

## Data
The dataset includes spike trains recorded from 98 neural units and the trajectory of the monkey's hand during 182 reaching movements across 8 different angles. For each trial, data consist of neural activity 300 ms before and 100 ms after movement, represented in 1 ms time steps. The training dataset, available on Blackboard as a `.mat` file, includes 100 trials per angle, while the test dataset consists of 82 trials per angle.

## Project Goals
- Develop a machine learning algorithm to predict hand position from neural data.
- The algorithm should estimate the X & Y coordinates of the hand, based on spike train data from the monkey's 98 neural units, across different time steps.
- Emphasis on real-time, causal decoding: the model should not use future data to predict current hand position.

## Competition Details
- Scoring: Root Mean Squared Error (RMSE) across both X & Y dimensions.
- Prizes for the team with the lowest error.
- Team participation: 3-4 members, with exceptions for groups of 5.
- Submission guidelines include algorithm codes and a final report.
- Final report structure: Maximum 4 pages, two-column A4 format, using the provided template on Blackboard.
- Algorithm must be original work, without reliance on external MATLAB toolboxes.
- Deadline: Algorithm due by 4 pm GMT, 19th March 2024; Report due by noon on the BMI exam day.

## Implementation and Submission
- MATLAB is the required platform for algorithm development and submission.
- Algorithms should adhere to the interface specifications provided in the MATLAB templates.
- Regular evaluations will be conducted to ensure the causality of the model.
- Submissions must be original and cite all resources used.

## Acknowledgment
The neural data for this project have been generously provided by the laboratory of Prof. Krishna Shenoy at Stanford University and are exclusively for educational purposes in this course.
