#!/usr/bin/env python

import sys
sys.path.append('../')

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

from args import args

if __name__=='__main__':
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} type (u|s|us)")
    sys.exit(1)

  if sys.argv[1] == 'u':
    

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
      'n_estimators': 400,
      'min_samples_leaf': 3,
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
      'max_bin': 16,
      'colsample_bytree': 0.5,
    }
  else:
    print("Allowed models: (lr|rf)")
    sys.exit(1)


  cohort = 'notes_all'
  data_df = pd.read_csv(f'data/{cohort}_proc.csv', usecols=['hadm_id', 'note', 'imi_adm_label'])
  data_df = data_df[data_df['imi_adm_label'] != -1].reset_index(drop=True)

  vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=60_000)
  args.workdir = Path('data/workdir')
  args.modeldir = args.workdir/f'{cohort}/{model_name}/models'
  threshold = args.threshold
  start_seed, n_iters = 127, 100

  clfs, targs, probs, preds = run_iters(data_df, clf_model, clf_params, vectorizer, threshold, start_seed, n_iters=n_iters)

  for i, clf in enumerate(clfs):
    pickle.dump(clf, open(args.modeldir/f'seed_{start_seed + i}.pkl', 'wb'))

  with open(args.workdir/f'{cohort}/{model_name}/preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(probs, f)
    pickle.dump(preds, f)
