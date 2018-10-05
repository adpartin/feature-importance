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

APP = 'nt'
DATAPATH = '/vol/ml/apartin/Benchmarks/Data/Pilot1'
FILENAME = 'matched_normal_samples.FPKM-UQ.log-transformed.csv'
GENEMAPFILE = 'lincs1000.tsv'
OUTDIR = os.path.join(file_path, 'data')
SEED = 0


def run():
    # Create necessary dirs
    utils.make_dir(OUTDIR)  # os.makedirs(OUTDIR, exist_ok=True)

    # Load data
    print('Loading NT data ...')
    nt = pd.read_csv(os.path.join(DATAPATH, FILENAME), sep=',')
    mm = pd.read_csv(os.path.join(DATAPATH, GENEMAPFILE), sep='\t')

    # Extract lincs cols from the whole dataset
    nt = nt[['Sample'] + mm['gdc'].tolist()]

    # Map lincs gene names and sort genes alphabetically
    col_mapping = {mm.loc[g, 'gdc']: mm.loc[g, 'symbol'] for g in range(mm.shape[0])}
    nt = nt.rename(columns=col_mapping)
    nt = nt[['Sample'] + sorted(nt.columns[1:].tolist())]
    
    # Shuffle and extract the target label
    nt = nt.sample(n=nt.shape[0], axis=0, replace=False, random_state=SEED).reset_index(drop=True)
    nt['Sample'] = nt['Sample'].map(lambda s: s.split('-')[-1]).values
    nt.rename(columns={'Sample': 'y'}, inplace=True)

    print(nt['y'].value_counts())
    nt.to_csv(os.path.join(OUTDIR, f'{APP}_data'), sep='\t', index=False)
    

def main():
    # args = init_params()
    # run(args)
    run()


if __name__ == '__main__':
    main()