#!/usr/bin/env python

import pickle
from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir = path/f'workdir/gbm'

args = Namespace(
  path=path,
  workdir=workdir,
  figdir=path/'workdir/figures',
  dataset_csv=path/'modelready_mm.csv',
  modeldir=workdir/'models',
  vectordir=path/'workdir/vectordir',
  str_cols_pkl=path/'str_cols.pkl',
  structured_threshold=0.49,
  unstructured_threshold=0.47,
  mm_threshold=0.45,
)
