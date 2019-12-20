#!/usr/bin/env python

from argparse import Namespace
from pathlib import Path

path = Path('../data')
workdir = path/'workdir'

args = Namespace(
  path=path,
  workdir=workdir,
  figdir=path/'figures',
  dataset_csv=path/'unstructured_proc.csv',
  threshold=0.31,
  )