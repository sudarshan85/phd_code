#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
model_name = 'gbm'
workdir = path/f'workdir'

args = Namespace(
  path=path,
  workdir=workdir,
  model=model_name,
  figdir=workdir/'figdir',
  vectordir=workdir/'vectordir',
  modeldir=workdir/model_name/'models',
  mimic_notes=Path('../mimic_data/notes_all_proc.csv'),
  mlh_notes=Path('../mlh_data/notes_all_proc.csv'),
  mimic_src_thresh=0.12,
  mlh_src_thresh=0.4,
  mlh_src_test_thresh=0.46,
  )