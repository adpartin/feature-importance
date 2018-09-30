from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

OUTDIR = os.path.join(file_path, 'data')
SEED = 0


def run():
    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    # Load data
    df_rna, meta = utils.load_lincs1000(dataset='combat', sources=['gdc'])

    # Specify col name of the target variable (cancer type)
    target_col_name = 'ctype'

    # Drop small classes
    min_class_size = 300
    df_rna, df_rna_small = utils.drop_samples_on_class_count(df=df_rna, y=meta[target_col_name],
                                                             min_class_size=min_class_size)
    df_rna, meta = utils.update_df_and_meta(df_rna, meta, on='Sample')
    print(f'\n{meta[target_col_name].value_counts()}')    

    # Balance classes
    class_size = min_class_size
    df_rna, y_out, dropped_classes = utils.balance_df(df=df_rna, y=meta[target_col_name],
                                                      class_size=class_size, seed=SEED)
    df_rna, meta = utils.update_df_and_meta(df_rna, meta, on='Sample')
    print(f'\n{meta[target_col_name].value_counts()}')

    # Create the class `other`
    # df_other = df_rna_small.sample(min_class_size, random_state=SEED)
    # df_rna = pd.concat([df_rna, df_other], axis=0)
    # df_rna, meta = utils.update_df_and_meta(df_rna, meta, on='Sample')
    # print(f'df_rna.shape {df_rna.shape}')
    # print(meta[target_col_name].value_counts())

    xdata = df_rna.iloc[:, 1:].copy()
    ydata = meta['ctype'].values
    ydata = LabelEncoder().fit_transform(ydata)

    # Permute data
    shuf_idx = np.random.permutation(df_rna.shape[0])
    xdata = xdata.iloc[shuf_idx].reset_index(drop=True)
    ydata = pd.Series(ydata[shuf_idx], name='y')

    # Preprocess
    xdata, idx = utils.drop_low_var_cols(xdata)

    # Concat
    df = pd.concat([pd.DataFrame(ydata), xdata], axis=1)
    print(df.iloc[:3, :4])

    # Save
    df.to_csv(os.path.join(OUTDIR, 'tc_data'), sep='\t', index=False)


def main():
    # args = init_params()
    # run(args)
    run()


if __name__ == '__main__':
    main()