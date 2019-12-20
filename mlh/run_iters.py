#!/usr/bin/env python

import sys
sys.path.append('../')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from utils.model_utils import run_iters

from lr.args import args as lr_args
from rf.args import args as rf_args
from gbm.args import args as gbm_args

if __name__=='__main__':
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} model (lr|rf|gbm)")
    sys.exit(1)

  model_name = sys.argv[1]

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

  data_df = pd.read_csv(f'data/notes_all_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
  data_df = data_df[data_df['imi_adm_label'] != -1].reset_index(drop=True)

  vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=60_000)
  args.workdir = Path(f'data/workdir/{model_name}')
  args.modeldir = args.workdir/'models'
  threshold = args.threshold
  start_seed, n_iters = 127, 100

  clfs, targs, probs, preds = run_iters(data_df, clf_model, clf_params, vectorizer, threshold, start_seed, n_iters=n_iters)

  for i, clf in enumerate(clfs):
    pickle.dump(clf, open(args.modeldir/f'seed_{start_seed + i}.pkl', 'wb'))

  with open(args.workdir/f'preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(probs, f)
    pickle.dump(preds, f)
