'''
The initial attempt at building an RNN with LSTM. Used as baseline as later discovered module performs much much better.
'''

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional


file_path = 'Data/sample_drink_set.txt'

def prep(file_path):

    # Open and store file created from clean_and_regularize.py
    text = open(file_path, 'rb').read().decode(encoding='utf-8')

    # Get the unique values in file
    vocab = sorted(set(text))

    # Sets a mapping of the unique characters to numbers/indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    # Sets the maximum sentence length for a single input in characters
    seq_length = 100

    # Create training examples/targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    return sequences, vocab, idx2char, char2idx

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    # Build a deep bidirectional LSTM RNN using sigmoid and softmax
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),
    tf.keras.layers.Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),
    tf.keras.layers.Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),
    tf.keras.layers.Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),
    tf.keras.layers.Bidirectional(LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')),    
    tf.keras.layers.Dense(vocab_size, activation='softmax')])
    return model

def loss(labels, logits):
    
    # Define loss function
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store results
    text_generated = []

    # Temperature setting to define randomness or creativity of text generated. Lower more predictable, higher more random.
    temperature = 0.2

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


if __name__ == '__main__':

    file_path = "drinks.txt"

    sequences, vocab, idx2char, char2idx = prep(file_path)

    # variable to hold input and target data
    dataset = sequences.map(split_input_target)

    # Buffer size to shuffle the dataset
    BUFFER_SIZE = 1000
    BATCH_SIZE = 64

    # Shuffle and put data into batches
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(vocab) # Length of the vocabulary in chars
    embedding_dim = len(vocab) # The embedding dimension
    rnn_units = 128  # Number of RNN units
    
    # Build the model
    model = build_model(
        vocab_size = len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    # Compile the model
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = 'Data/chckpts'
    
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # train and fit the model
    history = model.fit(dataset, epochs=20, callbacks=[checkpoint_callback])

    # Rebuild model using preexisting parameters with the last produced chckpt/weights
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    # Finally print out generated text
    print(generate_text(model, start_string=u"vodka"))
    