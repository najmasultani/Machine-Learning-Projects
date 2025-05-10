# NumPy Neural Network & PyTorch MLP — From-Scratch Implementation

This project showcases a complete feed-forward neural network built **from scratch using NumPy**, as well as a **PyTorch-based MLP** trained on the FashionMNIST dataset. It was developed as part of **ECE421: Introduction to Machine Learning (Fall 2024)** at the University of Toronto.

## 🚀 Key Features

### 🔷 Part 1: Custom NumPy Feed-Forward Neural Network
- Fully Connected Layers with ReLU and Softmax
- Batch training with Cross Entropy Loss
- Gradient computation via backpropagation
- Modular architecture with activations, layers, loss, and optimizer
- Visualization of training and validation accuracy/loss
- Trainable on the Iris dataset

### 🔶 Part 2: PyTorch MLP on FashionMNIST
- Uses `nn.Linear`, `nn.ReLU`, and `nn.CrossEntropyLoss`
- Achieves ≥82% validation accuracy
- Includes training and validation loss plots
- Implemented in `PA3.ipynb` (Colab notebook)

## 📁 File Structure
- `activations.py` – Implements ReLU, Softmax, Sigmoid, etc.
- `layers.py` – Fully connected layer with forward/backward logic
- `loss.py` – Cross-entropy loss with backprop
- `model.py` – Neural network model class (training, forward, backward)
- `util.py` – Dataset loader, logging, preprocessing tools
- `PA3.ipynb` – PyTorch implementation on FashionMNIST (Colab notebook)

## 🧪 Run Instructions (NumPy Part)
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Then, you can run training and testing via model.py and util.py (or via provided notebook)
```

## 📝 Author
Developed by Najma Sultani as part of ECE421 – Introduction to Machine Learning at University of Toronto, Fall 2024
