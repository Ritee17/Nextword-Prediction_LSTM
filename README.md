# 🎭 Next Word Prediction with LSTM & Streamlit

A Deep Learning project that predicts the next word in a sequence of text, trained on Shakespeare's *"Hamlet"*. This project utilizes **Long Short-Term Memory (LSTM)** networks for sequence modeling and is deployed as an interactive web application using **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 📌 Project Overview
The goal of this project is to develop a language model capable of understanding context and generating text in the style of Shakespeare. By analyzing sequences of words from *Hamlet*, the model learns to predict the most probable next word, demonstrating the power of RNNs in Natural Language Processing (NLP).
LIVE DEMO - https://nextword-predictionlstm-p62xzk3lsy7rv3lhkfalak.streamlit.app/

### Key Features
* **Deep Learning Model:** Stacked LSTM architecture (Embedding -> LSTM -> Dropout -> Dense).
* **Dataset:** Complete text of Shakespeare's *Hamlet*.
* **Visualization:** TensorBoard integration for tracking training loss and accuracy.
* **Web App:** Interactive UI built with Streamlit for real-time predictions.
* **Early Stopping:** Implemented to prevent overfitting during training.

## 📂 Project Structure
Here is the organization of the repository:

| File Name | Description |
| :--- | :--- |
| `app.py` | The main Streamlit application file for deployment. |
| `experiemnt s.ipynb` | Jupyter Notebook containing data preprocessing, model training, and evaluation code. |
| `hamlet.txt` | The raw text data source used for training. |
| `next_word_lstm.h5` | The trained Keras model saved in HDF5 format. |
| `tokenizer.pickle` | The tokenizer object used to convert text to sequences (saved for consistent inference). |
| `requiremen ts.txt` | List of Python dependencies required to run the project. |

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/next-word-prediction-lstm.git](https://github.com/your-username/next-word-prediction-lstm.git)
    cd next-word-prediction-lstm
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r "requirements.txt"
    ```
    *(Note: Ensure the filename matches exactly as it appears in the repo)*

## 🖥️ Usage

### Running the Web App
To launch the interactive Next Word Predictor:
```bash
streamlit run app.py

```

This will open the application in your default web browser (usually at `http://localhost:8501`).

### Training the Model

If you want to retrain the model or experiment with hyperparameters:

1. Open `experiemnt s.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells sequentially to preprocess data, train the LSTM, and save the new model files.
3. Monitor training progress using TensorBoard (if configured in the notebook).

## 🧠 Model Architecture

The model is built using **TensorFlow/Keras** with the following layers:

1. **Embedding Layer:** Converts word indices into dense vectors of fixed size.
2. **LSTM Layer 1:** 150 units, returns sequences.
3. **Dropout Layer:** 20% dropout to prevent overfitting.
4. **LSTM Layer 2:** 100 units.
5. **Dense Output Layer:** Softmax activation to predict the probability of the next word among all vocabulary words.

## 📊 Screenshots

### Streamlit Interface
<img width="1380" height="893" alt="image" src="https://github.com/user-attachments/assets/00c4da52-c08d-49a5-b1b5-b38286eca93b" />
