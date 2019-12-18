#!/usr/bin/env python

import pickle
from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir = path/f'workdir/lr'

args = Namespace(
  path=path,
  workdir=workdir,
  figdir=workdir/'figures',
  structured_csv=path/'modelready_structured.csv',
  unstructured_csv=path/'modelready_unstructured.csv',
  mm_csv=path/'modelready_mm.csv',
  modeldir=workdir/'models',
  vectordir=path/'workdir/vectordir',
  str_cols_pkl=path/'vitals_stats_cols.pkl',
)

# notes_common_vital_threshold=0.41,
# full_common_vital_threshold=0.43,
# notes_common_all_threshold=0.46,
# full_common_all_threshold=0.5,
