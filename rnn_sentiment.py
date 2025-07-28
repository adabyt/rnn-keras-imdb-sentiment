import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

print("-----0. Set up Environment -----")

model_directory = 'RNN_models'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

plot_directory = 'RNN_plots'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

print("-"*100)

print("-----1. Loading Data -----")

# imdb.load_data directly gives pre-tokenised reviews, where each review is a sequence of integers (word IDs). 
# The labels y_train, y_test are 0 for negative and 1 for positive.
vocab_size = 10000  # Limit to top 10,000 words
(X_train_full, y_train_full), (X_test, y_test) = imdb.load_data(num_words=vocab_size) # Only keep top 10,000 words) 

print(X_train_full.shape, "training samples")   # (25000,) training samples
print(X_test.shape, "testing samples")          # (25000,) testing samples
print(y_train_full.shape, "training labels")    # (25000,) training labels
print(y_test.shape, "testing labels")           # (25000,) testing labels

print(np.isnan(y_train_full).any())  # Check for null values in training labels; False
print(np.isnan(y_test).any())  # Check for null values in testing labels; False

# Plotting the distribution of labels
plt.figure(figsize=(8, 5))
sns.countplot(x=y_train_full, label='Training Labels')
sns.countplot(x=y_test, label='Testing Labels')
plt.title('Distribution of Labels in the Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend().remove()
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, "label_distribution.png"), dpi=300)
plt.show()  

print("-"*100)

print("-----2. Data Preprocessing -----")
# Reviews in the IMDb dataset (and most text datasets) have varying lengths. 
# Neural networks, especially fixed-input ones, need inputs of a uniform size. 
# Therefore, we need to pad (add zeros) or truncate sequences to a max_length.
max_review_length = 500
X_train_full = pad_sequences(X_train_full, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Displaying the shapes of the training and testing datasets
print(X_train_full.shape, "training samples")   # (25000, 500) training samples
print(X_test.shape, "testing samples")          # (25000, 500) testing samples

# Split the training data into training and validation sets
X_train, X_val = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_val = y_train_full[:-5000], y_train_full[-5000:]

# Displaying the shapes of the training and validation datasets
print(X_train.shape, "training samples")   # (20000, 500) training samples
print(X_val.shape, "validation samples")   # (5000, 500) validation samples
print(y_train.shape, "training labels")    # (20000,) training labels
print(y_val.shape, "validation labels")    # (5000,) validation labels

print("-"*100)

print("-----3. Model Definition -----")
# Embedding Layer: This will be your first layer after the input.
    # Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)
    # vocab_size will be 10000 if you used num_words=10000 in imdb.load_data().
    # embedding_dim is a hyperparameter (e.g., 128, 256).
    # input_length is the max_review_length.
# Recurrent Layer (Choose one):
    # LSTM(units=128) or GRU(units=128)
    # Start with return_sequences=False as it's a classification task.
    # Can experiment with adding Dropout directly to the LSTM or GRU layer as dropout and recurrent_dropout arguments, or add Dropout layers afterwards.
# Dense Layers:
    # One or more Dense layers with relu activation.
    # A final Dense layer for output:
        # For binary classification (like IMDb), Dense(1, activation='sigmoid') and loss='binary_crossentropy'.
        # If multi-class sentiment (e.g., positive, negative, neutral), Dense(num_classes, activation='softmax') and loss='sparse_categorical_crossentropy' (if labels are integers) or loss='categorical_crossentropy' (if labels are one-hot encoded).

embedding_dim = 128

lstm_model = Sequential()
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_review_length)
lstm_model.add(embedding_layer)
lstm_model.add(LSTM(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(1, activation='sigmoid'))  # For binary classification
# Alternatively, use GRU
# gru_model = Sequential()
# gru_model.add(embedding_layer)
# gru_model.add(GRU(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
# gru_model.add(Dense(64, activation='relu'))
# gru_model.add(Dropout(0.5))
# gru_model.add(Dense(1, activation='sigmoid'))  # For binary classification

# Build the model
lstm_model.build(input_shape=(None, max_review_length,))  # Build the model with input shape. Specify None for batch size otherwise output for the intial layer will be (500, 128) instead of (None, 500, 128)
# Alternatively, if using GRU, use gru_model.build(input_shape=(max_review_length,))

# Display the model summary    
print(lstm_model.summary())  # Display the model summary
# Alternatively, use gru_model.summary() if using GRU

"""
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ embedding (Embedding)                │ (None, 500, 128)            │       1,280,000 │    input dim (vocab_size) is 10000, output_dim is 128 ┃ 10000 * 128 = 1,280,000 ┃
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm (LSTM)                          │ (None, 128)                 │         131,584 │    4 * [(input_dim * units) + (units * units) + units] = 4 * [(128 * 128) + (128 * 128) + 128] = 131,584 ┃ 4 gates in a standard LSTM: Forget, Input, Candidate Cell State (sometimes called G_gate), and Output Gate ┃ input_dim * units: Weights connecting the current input to the gate | units * units: Weights connecting the previous hidden state (recurrent connection) to the gate | units: Bias terms for the gate
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │           8,256 │    128 * 64 + 64 = 8,256 ┃ 128 input units from LSTM to 64 output units in Dense layer ┃
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64)                  │               0 │    Dropout layer does not have parameters ┃
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │              65 │    64 * 1 + 1 = 65 ┃ 64 input units from Dense layer to 1 output unit ┃
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,419,905 (5.42 MB)
 Trainable params: 1,419,905 (5.42 MB)
 Non-trainable params: 0 (0.00 B)
None
"""


# Save the model architecture to a file
model_json = lstm_model.to_json()
with open(os.path.join(model_directory, "lstm_model.json"), "w") as json_file:
    json_file.write(model_json) 
# Alternatively, if using GRU, save gru_model to a file
# with open(os.path.join(model_directory, "gru_model.json"), "w") as json_file:
#     json_file.write(gru_model.to_json())  

# Save the model weights
lstm_model.save_weights(os.path.join(model_directory, "lstm_model.weights.h5"))
# If using GRU, save gru_model weights
# gru_model.save_weights(os.path.join(model_directory, "gru_model.weights.h5")) 

print("Model architecture and weights saved.")  

# Plotting the model architecture
plot_model(lstm_model, to_file=os.path.join(plot_directory, 'lstm_model_architecture.png'), show_shapes=True, show_layer_names=True)
# If using GRU, plot the GRU model architecture similarly
# plot_model(gru_model, to_file=os.path.join plot_directory, 'gru_model_architecture.png'), show_shapes=True, show_layer_names=True)


print("-"*100)

print("-----4. Model Compilation -----")
# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1
)

# Training the Model
EPOCHS = 50 
history = lstm_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping])

print("-"*100)

print("-----5. Model Evaluation -----")

score = lstm_model.evaluate(X_test, y_test, verbose=1)


final_epoch = history.epoch[-1]  # Get the last epoch number
print(f"Final Epoch: {final_epoch}, Test Loss: {score[0]:.4f}, Test Accuracy: {score[1]:.4f}")
# Final Epoch: 7, Test Loss: 0.5120, Test Accuracy: 0.8434

# Plotting the training and validation accuracy and loss
plt.figure(figsize=(12, 6)) 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.scatter(final_epoch, score[1], label='Test Accuracy', marker='o', color='red', s=100, zorder=5)
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(plot_directory, "model_accuracy.png"), dpi=300)
plt.show()  

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.scatter(final_epoch, score[0], label='Test Loss', marker='o', color='red', s=100, zorder=5)
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(plot_directory, "model_loss.png"), dpi=300)
plt.show()

"""
Based on the current set-up, the model achieves a test accuracy of approximately 84.34% after 7 epochs with a final test loss of around 0.5120.
This indicates that the model is performing reasonably well on the dataset, however, the graphs suggest potential overfitting, as the training accuracy is significantly higher than the validation and test accuracy, and the training loss is lower than the validation and test loss. 
Experimenting with the following could help improve the model's performance:
- Different embedding_dim values.
- LSTM vs. GRU.
- Adjust units in the recurrent layer.
- Add more Dense layers.
- Vary max_review_length.
- Adjust Dropout rates.
"""
