# ©Zizhe Wang, TU Dresden, 2021

# This script is used to augment fake data into real data. You have one label
# and two classe (e.g. label=npm1, two classes are: npm1=1 (true), npm1=0 (false)),
# fake data of the two classes can be added to the data for data augmentation.

# Basic idea of this script is: sort the .csv file based on labels, now one class
# of the data is at the bottom (e.g. npm1=0 are at the bottom), add x lines of
# fake data to bottom, then make the other class of data to the bottom and
# add x lines of fake data to bottom.

# Use the "make_csv.py" to generate .csv file for generated images

# import packages
import random
import pandas as pd
import numpy as np
import click
from sklearn.utils import shuffle

#----------------------------------------------------------------------------

# https://click.palletsprojects.com/en/8.0.x/
@click.command()

# arguments which need to be given
@click.option('--csv_input', help='Input csv', required=True, metavar='DIR')
@click.option('--csv_output', help='Output csv', required=True, metavar='DIR')
# feed the fake data with label 1 first
@click.option('--csv_fake_label_1', help='First input csv of fake data', required=True, metavar='DIR')
@click.option('--csv_fake_label_0', help='Second input csv of fake data', required=True, metavar='DIR')
@click.option('--label', help='Label name in csv', type=str)
@click.option('--num_label_1_real', help='Number of real data with label=1 used for training', type=int, metavar='INT')
@click.option('--num_label_0_real', help='Number of real data with label=0 used for training', type=int, metavar='INT')
@click.option('--num_label_1_total', help='Total number of data with label=1 (real data + fake data as augmentation )', type=int, metavar='INT')
@click.option('--num_label_0_total', help='Total number of data with label=0 (real data + fake data as augmentation)', type=int, metavar='INT')


#----------------------------------------------------------------------------

def main(
    csv_input, csv_output, csv_fake_label_1, csv_fake_label_0, label,
    num_label_1_real, num_label_0_real, num_label_1_total, num_label_0_total
    ):

    # pandas.read_csv
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    data = pd.read_csv(csv_input) # read the .csv which saves the original data
    # read the .csv which saves the fake data of class x
    data_fake_label_1 = pd.read_csv(csv_fake_label_1)
    # read the .csv which saves the fake data of class y
    data_fake_label_0 = pd.read_csv(csv_fake_label_0)

     # number of rows which will be extracted from 'csv_fake_label_0' and added to 'data'
    num_1 = num_label_0_total - num_labal_0_real

    len_data_fake_label_0 = len(data_fake_label_0.values) # get row length of 'csv_fake_label_0'
    scale_1 = list(range(0, len_data_fake_label_0)) # range in which random numbers will be generated

    # https://docs.python.org/3/library/random.html
    randoms_1 = random.sample(scale_1, num_1) # from length 'scale' generate 'num' random numbers

    columns_1 = data_fake_label_0.columns.values # get columns

    # np.vstack: Stack arrays in sequence vertically (row wise).
    # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    # remember: wenn extract some rows from fileX, use fileX.values[i] not fileX[2]
    for i in randoms_1:
        data = np.vstack((data, data_fake_label_0.values[i]))

    # convert to DataFrame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    new_data = pd.DataFrame(data, columns=columns_1)

    # shuffle 3 times
    new_data = shuffle(new_data)
    new_data = shuffle(new_data)
    new_data = shuffle(new_data)

    # --------------------------------------------------------------------------

    csv_sorted_2 = new_data.sort_values(by=[label], axis=0, ascending=True)

    num_2 = num_label_1_total - num_label_1_real

    len_data_fake_label_1 = len(data_fake_label_1.values) # get row length of 'csv_fake_label_0'
    scale_2 = list(range(0, len_data_fake_label_1)) # range in which random numbers will be generated

    # https://docs.python.org/3/library/random.html
    randoms_2 = random.sample(scale_2, num_2) # from length 'scale' generate 'num' random numbers

    columns_2 = data_fake_label_1.columns.values # get columns

    # np.vstack: Stack arrays in sequence vertically (row wise).
    # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    # remember: wenn extract some rows from fileX, use fileX.values[i] not fileX[2]
    for j in randoms_2:
        new_data = np.vstack((new_data, data_fake_label_1.values[j]))

    # convert to DataFrame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    final_data = pd.DataFrame(new_data, columns=columns_2)

    # shuffle 3 times
    final_data = shuffle(final_data)
    final_data = shuffle(final_data)
    final_data = shuffle(final_data)

    print('Done')

    # save to .csv as output
    # pandas.DataFrame.to_csv
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv
    # (path / name of output, row names (index) will not be written (otherwise there will be a new column to store row names))
    return final_data.to_csv(csv_output, index=False)

if __name__ == "__main__":
    main()
