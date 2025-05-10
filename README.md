# Machine Learning Projects

This repository showcases a growing collection of my machine learning projects, ranging from classic ML algorithms to advanced deep learning and applied research. These examples highlight my experience with classification, regression, optimization, neural networks, reinforcement learning, and bioinformatics. Projects are implemented using tools like **NumPy**, **scikit-learn**, **PyTorch**, **LangChain**, and **transformers**.

In addition to individual projects, this repository also references collaborative work such as **SmartStudy** (academic habit optimizer using GPT-4 and TabPFN) and **Cognify AI** (an AI-powered educational assistant for kids built with DeepPiXEL and UTESCA).

---

## Projects Included

### 1. Pocket Algorithm & Linear Regression
- Implemented binary classification using the Pocket Algorithm on the IRIS dataset  
- Linear regression with mean squared error on the Diabetes dataset  
- Compared performance with `scikit-learn` baselines  

### 2. Gradient Descent & Multiclass Logistic Regression
- Implemented four variants of stochastic gradient descent: **SGD**, **Heavy-ball**, **Nesterov**, and **Adam**  
- Built multiclass logistic regression from scratch with one-hot encoding and softmax  
- Compared optimizer performance on IRIS and Digits datasets  

### 3. Feedforward Neural Networks (NumPy + PyTorch)
- Implemented all layers of a 2-layer neural network using NumPy  
- Trained and tuned on the IRIS dataset using cross-entropy loss  
- Built and trained a PyTorch MLP on FashionMNIST using `nn.Linear`, `ReLU`, and `CrossEntropyLoss`  

### 4. Recurrent Neural Networks & Reinforcement Learning
- Vanilla RNN and LSTM for the Adding Problem (sequence regression)  
- Value Iteration and Q-Learning agents applied to **GridWorld**, **Crawler**, and **Pacman**  
- Approximate Q-Learning using feature-based representations  

### 5. Predicting Disease-Causing Mutations (Bioinformatics Hackathon 2024)
- Developed during a bioinformatics hackathon in Jan 2024  
- Built a machine learning model to predict whether a genetic sequence contains mutations related to **Alzheimer’s disease**, with a focus on **women's health**
- Used genomic mutation data from **DSS NIAGADS** and **GenomeKit**, filtered based on p-values < 0.05  
- Built features using biological mutation transformations, trained a **Random Forest** classifier combined with **Borzoi** from **gReLU** for mutation effect prediction  
- Challenges involved unfamiliarity with genomics and time-limited setup; overcame these through collaboration and rapid research  
- Inspired by the disproportionate impact of Alzheimer’s on women and the shortage of genetic prediction tools  
- Future improvements include increasing accessibility for clinicians and refining mutation localization  
- 🔗 [View Project Repository](https://devpost.com/software/predicing-disease-causing-mutations) <!-- Replace with actual URL -->

### 6. SmartStudy: Personalized Academic Recommendation System *(Collaborative Project)*
- A collaborative project focused on predicting and improving student GPA based on lifestyle and study patterns  
- Models used: **TabPFN**, **CatBoost**, **XGBoost**, **MLP**, **TabNet**, **1D CNN**, and stacked ensembles  
- Used **Bayesian Optimization** to recommend behavior changes that match students’ GPA goals  
- Integrated **K-Nearest Neighbors** for matching similar students and **GPT-4** to convert outputs into academic advice  
- Includes a **Gradio web app** for user interaction  
- [🔗 View Project Repository](https://github.com/elorie-bernard-lacroix/SmartStudy) <!-- Replace with actual URL -->

### 7. Cognify AI: Adaptive Learning Assistant for Kids *(Private Project)*
- Built as part of the **UTESCA x DeepPiXEL** consulting initiative  
- A GenAI solution for personalized education targeting children aged 8–12  
- I contributed as an **ML developer** focusing on integrating multiple AI models  
- Features include:
  - Text chat using **LangChain + GPT-3.5**  
  - Voice chat via **Whisper + GPT + gTTS**  
  - Object recognition via **Vision Transformer (ViT)**  
  - Handwriting-to-math solver using **EasyOCR** and **TrOCR**  
  - Real-time transcription and TTS  
  - Adaptive learning progress tracking  
  - Security features like encryption and authentication  
- The codebase is private due to project confidentiality.

---

## 🚀 Coming Soon
This repository will be expanded with more advanced machine learning projects. 
