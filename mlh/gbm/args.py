#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir = path/'workdir/gbm'

args = Namespace(
  path=path,
  workdir=workdir,
  figdir=path/'workdir/figures',
  vectordir=path/'workdir/vectordir',
  dataset_csv=path/'unstructured_proc.csv',
  threshold=0.11,
)