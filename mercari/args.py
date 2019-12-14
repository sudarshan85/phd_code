#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('data')
workdir = path/'workdir'

args = Namespace(
  path=path,
  workdir=workdir,
  train_tsv=path/'train.tsv',
  test_tsv=path/'test.tsv',
  test2_tsv=path/'test_stg2.tsv',
  modeldir=workdir/'models',
  vectordir=workdir/'vectordir',
  figdir=workdir/'figures',
)