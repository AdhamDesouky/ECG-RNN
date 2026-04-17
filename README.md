# Clinical ECG Classification: Arrhythmia Detection

## Objective
Develop a recurrent neural network (RNN) capable of identifying abnormal heartbeats in ECG signals. In a clinical setting, missing a sick patient is far more dangerous than a false alarm. Therefore, your primary goal is high **Sensitivity (Recall)**.

## Requirements
To pass the clinical safety validation, your final model must achieve:
- **Recall (Sensitivity):** > 90% on the official test set.
- **Architecture Coverage:** You must implement and compare three types of layers:
  1. `SimpleRNN`
  2. `LSTM`
  3. `GRU`

## Project Structure
- `assignment.ipynb`: Your primary workspace. Complete the `TODO` sections.
- `utils.py`: Contains the pre-defined evaluation logic. Do not modify this file.
- `requirements.txt`: List of required libraries.

## Evaluation
You will be evaluated on your ability to:
1. Explain the "Vanishing Gradient" problem in SimpleRNN.
2. Demonstrate how LSTM/GRU gates solve this problem using the "Cell State" highway.
3. Optimize the decision threshold to prioritize patient safety.
