# ©Zizhe Wang, TU Dresden, 2021

# import packages
import os
import pandas as pd
import click
from sklearn.utils import shuffle

# https://click.palletsprojects.com/en/8.0.x/
@click.command()

# arguments which need to be given
@click.option('--input_path', help='Input path / Folder name', required=True, metavar='DIR')
@click.option('--output_name', help='Output name', required=True, metavar='DIR')
@click.option('--label', help='Label name', type=str)
@click.option('--true_or_false', help='Label true or false', type=bool, metavar='BOOL')

# -----------------------------------------------------------------------------

def make_csv(input_path, output_name, label, true_or_false):

    # read image paths
    images = [image for image in os.listdir(input_path)]

    # create DataFrame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    df = pd.DataFrame()

    # set name and contents of first column
    df['img_path'] = [input_path+'/'+str(i) for i in images]

    # set name and contents of second column
    if true_or_false == True:
        df[label]='1'
    else:
        df[label]='0'

    # shuffle 3 times
    df = shuffle(df)
    df = shuffle(df)
    df = shuffle(df)

    print ('Done')

    # save to .csv as output
    # pandas.DataFrame.to_csv
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html?highlight=to_csv
    # (path / name of output, row names (index) will not be written (otherwise there will be a new column to store row names))
    return df.to_csv(output_name, index=False)

if __name__ == "__main__":
    make_csv()
