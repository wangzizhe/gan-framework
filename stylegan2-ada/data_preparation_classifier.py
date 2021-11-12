# ©Zizhe Wang, TU Dresden, 2021
# using command example:
# python data_preparation_classifier.py --csv_input=./test_train.csv --csv_output=./test_output.csv --csv_fake_label_1=fake_data_label_1.csv --csv_fake_label_0=fake_data_label_0.csv --propotion=0.2 --label=npm1 --num_label_1=300 --num_label_0=740

import random
import pandas as pd
import numpy as np
import click
from sklearn.utils import shuffle

#----------------------------------------------------------------------------

# https://click.palletsprojects.com/en/8.0.x/
@click.command()

@click.option('--csv_input', help='Input csv', required=True, metavar='DIR')
@click.option('--csv_output', help='Output csv', required=True, metavar='DIR')
# feed the fake data with label 1 first
@click.option('--csv_fake_label_1', help='First input csv of fake data', required=True, metavar='DIR')
@click.option('--csv_fake_label_0', help='Second input csv of fake data', required=True, metavar='DIR')
@click.option('--propotion', help='Propotion of fake data', type=float)
@click.option('--label', help='Label name in csv', type=str)
@click.option('--num_label_1', help='Number of data with label=1', type=int, metavar='INT')
@click.option('--num_label_0', help='Number of data with label=0', type=int, metavar='INT')

#----------------------------------------------------------------------------

def main(
    csv_input, csv_output, csv_fake_label_1, csv_fake_label_0,
    propotion, label, num_label_1, num_label_0
    ):

    # pandas.read_csv
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    data = pd.read_csv(csv_input) # read the .csv which saves the original data
    # read the .csv which saves the fake data of class x
    data_fake_label_1 = pd.read_csv(csv_fake_label_1)
    # read the .csv which saves the fake data of class y
    data_fake_label_0 = pd.read_csv(csv_fake_label_0)

    # pandas.DataFrame.sort_values
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    # (column label="npm1", sorted by column label, sort descending)
    csv_sorted = data.sort_values(by=[label], axis=0, ascending=False)

     # number of rows which will be extracted from 'csv_fake_label_0' and added to 'data'
    num_1 = round(num_label_0*propotion)

    dropped_data_1 = csv_sorted.drop(csv_sorted.tail(num_1).index) # drop n rows from tail of sorted 'data'

    len_data_fake_label_0 = len(data_fake_label_0.values) # get row length of 'csv_fake_label_0'
    scale_1 = list(range(0, len_data_fake_label_0)) # range in which random numbers will be generated

    # https://docs.python.org/3/library/random.html
    randoms_1 = random.sample(scale_1, num_1) # from length 'scale' generate 'num' random numbers

    columns_1 = data_fake_label_0.columns.values # get columns

    # np.vstack: Stack arrays in sequence vertically (row wise).
    # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    # remember: wenn extract some rows from fileX, use fileX.values[i] not fileX[2]
    for i in randoms_1:
        dropped_data_1 = np.vstack((dropped_data_1, data_fake_label_0.values[i]))

    # convert to DataFrame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    new_data = pd.DataFrame(dropped_data_1, columns=columns_1)

    # shuffle 3 times
    new_data = shuffle(new_data)
    new_data = shuffle(new_data)
    new_data = shuffle(new_data)

    # --------------------------------------------------------------------------

    csv_sorted_2 = new_data.sort_values(by=[label], axis=0, ascending=True)

    num_2 = round(num_label_1*propotion)

    dropped_data_2 = csv_sorted_2.drop(csv_sorted_2.tail(num_2).index)

    len_data_fake_label_1 = len(data_fake_label_1.values) # get row length of 'csv_fake_label_0'
    scale_2 = list(range(0, len_data_fake_label_1)) # range in which random numbers will be generated

    # https://docs.python.org/3/library/random.html
    randoms_2 = random.sample(scale_2, num_2) # from length 'scale' generate 'num' random numbers

    columns_2 = data_fake_label_1.columns.values # get columns

    # np.vstack: Stack arrays in sequence vertically (row wise).
    # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    # remember: wenn extract some rows from fileX, use fileX.values[i] not fileX[2]
    for j in randoms_2:
        dropped_data_2 = np.vstack((dropped_data_2, data_fake_label_1.values[j]))

    # convert to DataFrame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    final_data = pd.DataFrame(dropped_data_2, columns=columns_2)

    # shuffle 3 times
    final_data = shuffle(final_data)
    final_data = shuffle(final_data)
    final_data = shuffle(final_data)

    print('Done')

    # pandas.DataFrame.to_csv
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv
    # (path / name of output, row names (index) will not be written (otherwise there will be a new column to store row names))
    return final_data.to_csv(csv_output, index=False)

if __name__ == "__main__":
    main()
