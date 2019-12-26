#!/usr/bin/env python

import sys
sys.path.append('../')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import pickle

from tqdm import trange
import numpy as np
import scipy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from utils.data_utils import set_group_splits

from lr.args import args as lr_args
from rf.args import args as rf_args
from gbm.args import args as gbm_args

if __name__=='__main__':
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} model (lr|rf|gbm) transfer_from (mimic|mlh)")
    sys.exit(1)

  model_name = sys.argv[1]
  transfer_from = sys.argv[2]

  if model_name == 'lr':
    args = lr_args
    clf_model = LogisticRegression
    clf_params = {
      'class_weight': 'balanced',
    }
  elif model_name == 'rf':
    args = rf_args
    clf_model = RandomForestClassifier
    clf_params = {
      'n_estimators': 10,
      'min_samples_leaf': 5,
      'oob_score': True,
      'class_weight': 'balanced',
      'n_jobs': -1,
    }
  elif model_name == 'gbm':
    args = gbm_args
    clf_model = lgb.LGBMClassifier
    clf_params = {
      'objective': 'binary',
      'metric': 'binary_logloss',  
      'is_unbalance': True,
      'learning_rate': 0.05,
      'num_leaves': 200,
      'boosting_type': 'dart',
      'subsample_freq': 5,
      'subsample': 0.7,
      'colsample_bytree': 0.5,
    }
  else:
    print("Allowed models: (lr|rf|gbm)")
    sys.exit(1)

  print("Loading data...")

  if transfer_from == 'mimic':
    transfer_to = 'mlh'
    source_df = pd.read_csv('mimic_data/notes_all_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
    source_df = source_df[source_df['imi_adm_label'] != -1].reset_index(drop=True)
    target_df = pd.read_csv('mlh_data/notes_all_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
    target_df = target_df[target_df['imi_adm_label'] != -1].reset_index(drop=True)
    vec_name = 'mimic2mlh'
    threshold = args.mimic_src_thresh
  elif transfer_from == 'mlh':
    transfer_to = 'mimic'
    source_df = pd.read_csv('mlh_data/notes_all_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
    source_df = source_df[source_df['imi_adm_label'] != -1].reset_index(drop=True)
    target_df = pd.read_csv('mimic_data/notes_all_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
    target_df = target_df[target_df['imi_adm_label'] != -1].reset_index(drop=True)
    vec_name = 'mlh2mimic'
    threshold = args.mlh_src_thresh
  else:
    print("Allowed transfer: (mimic|mlh)")
    sys.exit(1)

  path = Path('data/')
  workdir = path/'workdir'
  vectordir = workdir/'vectordir'
  modeldir = workdir/f'{model_name}/models'

  print(f"Transferring from {transfer_from} to {transfer_to} with model {model_name} with threshold {threshold}")
  source_mdl = pickle.load(open(modeldir/f'{transfer_from}_full.pkl', 'rb'))

  with open(vectordir/f'{vec_name}.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
  
  clfs, targs, preds, probs = [], [], [], []
  start_seed, n_iters = 127, 100
  t = trange(start_seed, start_seed + n_iters, desc='Run #', leave=True)

  for seed in t:
    t.set_description(f"Run # (seed {seed})")
    df = set_group_splits(target_df.copy(), group_col='hadm_id', seed=seed, pct=0.9)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    x_train = vectorizer.transform(train_df['note'])
    x_test = vectorizer.transform(test_df['note'])

    y_train, y_test = train_df['imi_adm_label'], test_df['imi_adm_label']
    targs.append(y_test)

    source_mdl.fit(x_train, y_train)
    clfs.append(source_mdl)

    prob = source_mdl.predict_proba(x_test)
    probs.append(prob)

    y_pred = (prob[:, 1] > threshold).astype(np.int64)
    preds.append(y_pred)

  for i, clf in enumerate(clfs):
    pickle.dump(clf, open(modeldir/f'{transfer_from}_{transfer_to}_seed_{start_seed + i}.pkl', 'wb'))

  with open(workdir/f'{model_name}/{transfer_from}_{transfer_to}_preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(probs, f)
    pickle.dump(preds, f)
