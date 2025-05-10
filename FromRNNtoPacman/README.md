# RNN & Reinforcement Learning 
This project implements a Recurrent Neural Network (RNN) for sequence learning and Reinforcement Learning (RL) agents for decision-making in simulated environments. Built as part of **ECE421: Introduction to Machine Learning (Fall 2024)** at the University of Toronto.

---

## 🔁 Part 1: Sequence Learning with RNNs and LSTM (PyTorch)

### Task: Adding Problem
Trained models to predict the sum of two highlighted values in a randomly generated sequence using:
- **Vanilla RNN**
- **LSTM**

### Highlights:
- Implemented in `models.py` and `train.py`
- Dataset from `make_dataset.py`
- Trained on sequences of lengths 10, 25, and 50
- Hyperparameter tuning (learning rate, hidden size)
- Results visualized in provided notebook `PA4.ipynb`

---

## 🎮 Part 2: Reinforcement Learning in Gridworld and Pacman (Berkeley AI)

### Implemented Agents:
- `ValueIterationAgent` (Value Iteration for MDPs)
- `QLearningAgent` (Model-free Q-learning)
- `PacmanQAgent` (RL applied to Pacman)
- `ApproximateQAgent` (Feature-based Q-learning)

### Highlights:
- Solved Gridworld, Crawler, and Pacman tasks
- Performed policy synthesis and Q-value analysis
- Handled noisy environments and reward shaping
- Implemented `epsilon`-greedy strategies and feature extractors

### Edited Files:
- `valueIterationAgents.py`
- `qlearningAgents.py`
- `analysis.py`
- `train.py`
- `models.py`

---

## 📁 Directory Structure

```bash
├── models.py                 # RNN & LSTM model logic
├── train.py                 # Training loop for sequence learning
├── analysis.py              # Answers for RL policy tuning
├── qlearningAgents.py       # Q-learning implementations
├── valueIterationAgents.py  # Value iteration agent
├── PA4.ipynb                # Notebook for training & plots (Colab)
├── requirements.txt         # All necessary Python packages
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📝 Author

Developed by Najma Sultani as part of ECE421 – Introduction to Machine Learning at University of Toronto, Fall 2024
Includes RL components adapted from UC Berkeley’s CS188.

