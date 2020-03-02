'''
Run this file in order to run your own textgenrnn model and perform text generation. Setting I used are already coded into 
model_cfg and train_cfg. Must first install textgenrnn on your computer by running, !pip install textgenrnn.
'''

from textgenrnn import textgenrnn

def textgen(file_name, model_name):

    model_cfg = {
        'word_level': False,   # set to True if want to train a word-level model (requires more data and smaller max_length)
        'rnn_size': 128,   # number of LSTM cells of each layer
        'rnn_layers': 5,   # number of LSTM layers
        'rnn_bidirectional': True,   # consider text both forwards and backward, can give a training boost
        'max_length': 40,   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
        'max_words': 10000,   # maximum number of words to model; the rest will be ignored (word-level model only)
    }

    train_cfg = {
        'line_delimited': True,   # set to True if each text has its own line in the source file
        'num_epochs': 40,   # set higher to train the model for longer
        'gen_epochs': 1,   # generates sample text from model after given number of epochs
        'train_size': 0.8,   # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
        'dropout': 0.2,   # ignore a random proportion of source tokens each epoch, allowing model to generalize better
        'validation': False,   # If train__size < 1.0, test on holdout dataset; will make overall training slower
        'is_csv': False   # set to True if file is a CSV exported from Excel/BigQuery/pandas
    }

    # This instantiates the module 
    textgen = textgenrnn(name=model_name)

    # Sets textgen model to have all parameters previously specified and take data from preprocessed 
    # file created in clean_and_regularize.py
    train_function = textgen.train_from_file 

    train_function(
        file_path=file_name,
        new_model=True,
        num_epochs=train_cfg['num_epochs'],
        gen_epochs=train_cfg['gen_epochs'],
        batch_size=1024,
        train_size=train_cfg['train_size'],
        dropout=train_cfg['dropout'],
        validation=train_cfg['validation'],
        is_csv=train_cfg['is_csv'],
        rnn_layers=model_cfg['rnn_layers'],
        rnn_size=model_cfg['rnn_size'],
        rnn_bidirectional=model_cfg['rnn_bidirectional'],
        max_length=model_cfg['max_length'],
        dim_embeddings=100,
        word_level=model_cfg['word_level'])

if __name__ == '__main__':

    file_name = "drinks.txt"
    model_name = 'drinks_model'

    textgen(file_name, model_name)