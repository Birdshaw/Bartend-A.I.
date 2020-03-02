'''
Run script to clean and regularize dataset then write results to a text file
'''

import pandas as pd

def load_df(path):

    # Load csv to a pandas dataframe
    df = pd.read_csv('Data/drinksandcocktails_1.0.9.csv', escapechar='\\')

    # Filter out only the category of drinks desired i.e. cocktails
    df = df[(df['d_cat'] == 'Ordinary Drink') | (df['d_cat'] == 'Cocktail')]

    # Drop null rows
    df = df.dropna(axis='index')

    # Reformate data in d_ingredients column and place in new column
    df['ingredients'] = (df['d_ingredients'].str.split('|').str.join(', ').tolist())

    return df

def replace_nums(df):

    # A list of all unique units of measurments and fractions in order to replace and regularize all to mls

    measurements = {'1/2': '0.5','1/4': '0.25', '3/10': '0.3','2/10': '0.2','1/10': '0.1','1/8': '0.125','1/3': '0.3','2/3': '0.6',
                    '3/4': '0.75','1\s0.5': '1.5','1\s0.25': '1.25','1\s0.75': '1.75', '2\s0.25': '2.25', '2\s0.3':'2.3',  '2\s0.5': '2.5',
                   '2\s0.75': '2.75','1-0.5': '1.5', '2 .6': '2.6','3/4': '0.75','0.5-1|0.5 - 1': '0.75','1-2': '1.5','1 or 2': '1.5',
                    '1.5 - 2': '1.75', '1-2|1 - 2': '1.5','1-1.5|1 - 1.5': '1.25','4-6': '5','6-7|6 - 7': '6.5','2-3|2 - 3': '2.5',
                    '2.5-3|2.5 - 3': '2.75','3-4|3 - 4': '3.5','4-5|4 - 5': '4.5','5-7|5 - 7': '6','2-4|2 - 4': '3','8-12|8 - 12': '10',
                    '7-8|7 - 8': '7.5','1-3|1 - 3': '2','6-8|6 - 8': '7','8-10|8 - 10':'9','10-12|10 - 12': '11','6-10|6 - 10': '8',
                    '1.5-2|1.5 - 2': '1.75','3-6|3 - 6': '4.5','3-5|3 - 5': '4','5-6|5 - 6': '5.5','9-10|9 - 10': '9.5',
                    '5-10|5 - 10': '7.5','12-14|12 - 14': '13','.75-3/2|.75 - 3/2': '1.125','30-45': '40','cups': 'cup','pints': 'pint',
                    'quarts': 'quart','gallons': 'gallon', 'tblsp': 'tbsp'}


    # Run replacement twice in order to replace various fractions that such as 1/2 to 0.5 and then from 1\s0.5 to 1.5
    df['ingredients'] = df['ingredients'].replace(regex = measurements)

    df['ingredients'] = df['ingredients'].replace(regex = measurements)

    return df

def conversions(row):

    # Apply this function to take cleaned dataset and properly convert all measurements to ml.

    units = [['oz', 30.], ['gallon', 3785], ['shot', 44], ['liter', 1000], ['quart', 946], ['pint', 473], ['cup', 240], ['jigger', 44], ['tbsp', 18], ['tsp', 6], ['dash', 1], ['cl', 10]]
    num = [0.1, 0.125, 0.2, 0.25, 0.3, 0.5, 0.6, 0.75, 0.8, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.3, 2.5, 2.6, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7,
           7.5, 8, 9, 9.5, 10, 11, 12, 13, 16, 30, 40]

    for i in units:
        for j in num:
            val = float(j) * float(i[1])
            row['ingredients'] = row['ingredients'].replace(f"{j} {i[0]}", f"{val} ml")
    return row

def apply_then_to_text(df, txt_file_name):

    # Apply conversions to all rows of dataset
    df = df.apply(conversions, axis=1)

    # Take only the names and ingredients of cocktails to be used in RNN
    df_txt = df.loc[:, ['d_name', 'ingredients']]

    # Write these to a text file in order to be used by RNN
    df_txt.to_csv(txt_file_name, encoding='utf-8', header=None, index=False)

if __name__ == '__main__':

    path = 'Data/drinksandcocktails_1.0.9.csv'

    df = load_df(path)
    df = replace_nums(df)
    apply_then_to_text(df, "drinks.txt")
