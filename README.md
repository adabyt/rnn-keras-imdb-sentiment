# 📚 Sentiment Analysis on IMDb Reviews with RNNs (LSTM)

This project demonstrates how to build a **Recurrent Neural Network (RNN)** using **TensorFlow/Keras** to perform **sentiment analysis** on the IMDb movie review dataset. The model predicts whether a given movie review is **positive** or **negative** based on its text.

---

## 🧠 What are RNNs?

Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle **sequential data**, such as text, speech, or time series, by maintaining a “memory” of previous inputs through **recurrent connections**.  

Unlike traditional feedforward neural networks, RNNs pass information from one time step to the next, allowing the network to “remember” past context.  

Two popular RNN variants used in NLP are:
- **LSTM (Long Short-Term Memory):** Handles long-term dependencies by using gates to control what information is kept or discarded.
- **GRU (Gated Recurrent Unit):** A simplified version of LSTM with fewer gates but comparable performance in many tasks.

In this project, we use an **LSTM** layer to capture the sentiment context of movie reviews.

---

## 📂 Project Structure

```
rnn-keras-imdb-sentiment/
│
├── rnn_sentiment.py              # Main training script
├── RNN_models/                   # Saved model architecture & weights
│   ├── lstm_model.json
│   └── lstm_model.weights.h5
├── RNN_plots/                    # Accuracy/Loss plots & model diagram
│   ├── label_distribution.png
|   ├── lstm_model_architecture.png
│   ├── model_accuracy.png
│   └── model_loss.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How It Works

1️⃣ **Load Data:**  
- Use the IMDb dataset from Keras (`keras.datasets.imdb`).
- Keep the top 10,000 most frequent words for vocabulary efficiency.  

2️⃣ **Preprocess Data:**  
- Pad or truncate reviews to a maximum length of 500 words for uniform input size.

3️⃣ **Build the Model:**  
- **Embedding Layer:** Convert word indices into dense vector representations.  
- **LSTM Layer:** Capture sequence dependencies to understand sentiment context.  
- **Dense Layers:** Process LSTM outputs and map to a binary classification (positive or negative).

4️⃣ **Train & Evaluate:**  
- Use **Adam optimiser** and **binary crossentropy** loss.
- Train for up to **50 epochs** with **early stopping** if validation loss doesn’t improve for 5 epochs.
- Evaluate performance on the test set.

---

## 📊 Results

✅ **Final Epoch:** 7  
✅ **Test Accuracy:** ~84.34%  
✅ **Test Loss:** ~0.5120  

📉 Training accuracy exceeded validation accuracy, and training loss dipped below validation/test loss: suggests **potential overfitting**.

---


## 📈 Visualisations

The project generates:
- **Label Distribution Plot**
- **Training & Validation Accuracy Plot**  
- **Training & Validation Loss Plot**  
- **Model Architecture Diagram**

All saved in the `RNN_plots/` directory.

---

## 🔍 Conclusions

The model performs **reasonably well** on the IMDb dataset with an accuracy of around **84%**. However, there are signs of **overfitting**, as shown by the divergence between training and validation performance.

### 🔧 Future Improvements
- Experiment with different **embedding dimensions** (e.g., 64, 256).
- Compare **LSTM vs. GRU** performance (*code included for GRU*).
- Adjust the **number of units** in the recurrent layer.
- Add **more Dense layers** for deeper representations.
- Modify **max_review_length** to see how review truncation impacts results.
- Tune **Dropout rates** for better regularisation.


---

## 📜 License

MIT License – feel free to use and modify this code for your own projects.
