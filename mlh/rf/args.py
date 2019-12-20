#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
model_name='rf'
workdir = path/f'workdir'

args = Namespace(
  path=path,
  workdir=workdir,
  model=model_name,
  figdir=workdir/'figdir',
  dataset_csv=path/'notes_all_proc.csv',
  threshold=0.16,
  )