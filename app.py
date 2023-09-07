import tensorflow as tf

import numpy as np
import time

import json
with open("data/idxs_lookup_config.json", "r") as f:
    idxs_from_chars_cfg = json.load(f)
    
idxs_from_chars = tf.keras.layers.StringLookup(**idxs_from_chars_cfg)
chars_from_idxs = tf.keras.layers.StringLookup(
    vocabulary = idxs_from_chars.get_vocabulary(), invert=True, mask_token=None
)

# Batch size
BATCH_SIZE = 64

VOCAB_SIZE = len(idxs_from_chars.get_vocabulary())  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

# Build model
class KieuModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm_0 = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense_0 = tf.keras.layers.Dense(1024)
        self.dropout_0 = tf.keras.layers.Dropout(0.3)
        self.dense_1 = tf.keras.layers.Dense(512)
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dense_2 = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        # LSTM 1
        if states == None:
            state_h, state_c = self.lstm_0.get_initial_state(x)
            states = [state_h, state_c]
        x, state_h, state_c = self.lstm_0(x, initial_state=states, training=training)
        # FC
        x = self.dense_0(x, training=training)
        x = self.dropout_0(x)
        x = self.dense_1(x, training=training)
        x = self.dropout_1(x)
        x = self.dense_2(x, training=training)

        if return_state:
            return x, [state_h, state_c]
        else:
            return x

model = KieuModel(
    vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS
)
model.load_weights("/model/kieu_gpt")

# Text Generator
class PoemTeller(tf.keras.Model):
    def __init__(self, model, chars_from_idxs, idxs_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_idxs = chars_from_idxs
        self.idxs_from_chars = idxs_from_chars

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.idxs_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_idxs(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

teller = PoemTeller(model, chars_from_idxs, idxs_from_chars, 0.1)

inp = input("Starting string: ")
lines = int(input("Number of line"))

start = time.time()
states = None
next_char = tf.constant([inp])
result = [next_char]

i=0
while i < lines:
    next_char, states = teller.generate_one_step(next_char, states=states)
    result.append(next_char)
    if next_char[0].numpy().decode('utf-8') == '\n':
        i += 1

result = tf.strings.join(result)
output_text = result[0].numpy().decode('utf-8')
end = time.time()
print(output_text, '\n\n' + '-'*80)
print('Run time:', end - start)