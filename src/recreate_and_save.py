'''
Takes in the weights, vocab, and config files that are created when training a new textgenrnn model and allows for immediate
text generation based on that models training. Finally, saves the data to a text file for future use.
'''

from textgenrnn import textgenrnn

def recreate_and_save():

    w_path = 'Data/full_drinks_char_weights.hdf5' # Model weights.
    v_path = 'Data/full_drinks_char_vocab.json' # Model vocabulary.
    c_path = 'Data/full_drinks_char_config.json' # Model configurations.
    d_path = 'py_test.txt' # Recreated models text generation results destination.

    num_of_recipes = 20 
    temp = 0.2 # Controls how "creative" you desire these new recipes to be.

    textgen_recipe_gen = textgenrnn(weights_path = w_path, 
                                    vocab_path = v_path, 
                                    config_path = c_path
                                    )

    textgen_recipe_gen.generate_to_file(n = num_of_recipes, temperature = temp, destination_path = d_path)

if __name__ == '__main__':

    file_name = "drinks.txt"
    model_name = 'drinks_model'

    recreate_and_save()