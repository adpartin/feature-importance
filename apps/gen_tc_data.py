from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

file_path = os.path.dirname(os.path.relpath(__file__))

utils_path = os.path.abspath(os.path.join(file_path, '..', '..', 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

pfi_path = os.path.abspath(os.path.join(file_path, '..', 'pfi'))
sys.path.append(pfi_path)
import pfi
import pfi_utils

import warnings
warnings.filterwarnings('ignore')

APP = 'tc'
OUTDIR = os.path.join(file_path, 'data')
SEED = 0


def run():
    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    # Load data
    dataset = 'raw'
    df_rna, meta = utils.load_lincs1000(dataset=dataset, sources=['gdc'])

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

    print(f'\ndf_rna.shape {df_rna.shape}')
    print(f'meta.shape   {meta.shape}')

    # Create the class `other`
    # df_other = df_rna_small.sample(min_class_size, random_state=SEED)
    # df_rna = pd.concat([df_rna, df_other], axis=0)
    # df_rna, meta = utils.update_df_and_meta(df_rna, meta, on='Sample')
    # print(f'df_rna.shape {df_rna.shape}')
    # print(meta[target_col_name].value_counts())

    # Encode target variable
    ydata = meta['ctype'].values
    y_enc = LabelEncoder()
    ydata = y_enc.fit_transform(ydata)
    y_enc = pd.DataFrame(data={'label': np.arange(0, len(y_enc.classes_)), 'type': y_enc.classes_})
    y_enc.to_csv(os.path.join(OUTDIR, f'{APP}_y_enc'), sep='\t', index=False)

    # Permute data
    xdata = df_rna.iloc[:, 1:].copy()
    shuf_idx = np.random.permutation(xdata.shape[0])
    xdata = xdata.iloc[shuf_idx].reset_index(drop=True)
    ydata = pd.Series(ydata[shuf_idx], name='y')
    features = xdata.columns

    # Drop low var cols
    xdata, idx = utils.drop_low_var_cols(xdata, verbose=True)
    features = xdata.columns    

    # Split train/val
    xtr, xvl, ytr, yvl = train_test_split(xdata, ydata, test_size=0.2, random_state=SEED, shuffle=True, stratify=ydata)
    # print(xtr.index[:5])
    # print(ytr.index[:5])
    xtr, ytr = xtr.reset_index(drop=True), ytr.reset_index(drop=True)
    xvl, yvl = xvl.reset_index(drop=True), yvl.reset_index(drop=True)

    # Scale data
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xvl = scaler.transform(xvl)
    xtr = pd.DataFrame(xtr, columns=features)
    xvl = pd.DataFrame(xvl, columns=features)
    # print('xtr.var(axis=0).mean()', xtr.var(axis=0).mean())
    # print('xvl.var(axis=0).mean()', xvl.var(axis=0).mean())

    # Concat
    # data = pd.concat([pd.DataFrame(ydata), xdata], axis=1)
    data_train = pd.concat([pd.DataFrame(ytr), xtr], axis=1)
    data_val = pd.concat([pd.DataFrame(yvl), xvl], axis=1)
    print(f'\ndata_train.shape {data_train.shape}')
    print(f'data_val.shape   {data_val.shape}')

    # Save
    # data.to_csv(os.path.join(OUTDIR, f'{APP}_data'), sep='\t', index=False)
    data_train.to_csv(os.path.join(OUTDIR, f'{APP}_data_train_{dataset}'), sep='\t', index=False)
    data_val.to_csv(os.path.join(OUTDIR, f'{APP}_data_val_{dataset}'), sep='\t', index=False)


def main():
    # args = init_params()
    # run(args)
    run()


if __name__ == '__main__':
    main()