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
# from sklearn.ensemble import RandomForestClassifier
# import lightgbm as lgb

from utils.data_utils import set_group_splits

from lr.args import args as lr_args
# from rf.full_common_args import args as rf_args
# from gbm.full_common_args import args as gbm_args

if __name__=='__main__':
  if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} model (lr|rf|gbm) subset (s|u|u+s)")
    sys.exit(1)

  model_name = sys.argv[1]
  subset = sys.argv[2]

  if model_name == 'lr':
    args = lr_args
    clf_model = LogisticRegression
    clf_params = {
      'class_weight': 'balanced',
    }
  # elif model_name == 'rf':
  #   args = rf_args
  #   clf_model = RandomForestClassifier
  #   clf_params = {
  #     'n_estimators': 400,
  #     'min_samples_leaf': 3,
  #     'oob_score': True,
  #     'class_weight': 'balanced',
  #     'n_jobs': -1,
  #   }
  # elif model_name == 'gbm':
  #   args = gbm_args
  #   clf_model = lgb.LGBMClassifier
  #   clf_params = {
  #       'objective': 'binary',
  #     'metric': 'binary_logloss',
  #     'is_unbalance': True,
  #     'learning_rate': 0.05,
  #     'max_bin': 16,
  #     'colsample_bytree': 0.5,
  #   }
  else:
    print("Allowed models: (lr|rf|gbm)")
    sys.exit(1)

  str_cols = pickle.load(open('data/str_cols.pkl', 'rb'))
  cols = ['hadm_id'] + str_cols + ['note', 'imi_adm_label']    

  data_df = pd.read_csv('data/modelready_mm.csv', usecols=cols)
  data_df = data_df[data_df['imi_adm_label'] != -1].reset_index(drop=True)

  if subset == 's':
    print("Loading sturctured data...")
    threshold = args.structured_threshold
    data_df = data_df[['hadm_id'] + str_cols + ['imi_adm_label']].copy().reset_index(drop=True)
  elif subset == 'u':
    print("Loading unsturctured data...")
    data_df = data_df[['hadm_id', 'note', 'imi_adm_label']].copy().reset_index(drop=True)
    threshold = args.unstructured_threshold
  elif subset == 'u+s':    
    print("Loading multimodal data...")
    threshold = args.mm_threshold
  else:
    print("Allowed subsets (s|u|u+s)")
    sys.exit()

  print(data_df.shape)


  clfs, targs, preds, probs = [], [], [], []
  start_seed, n_iters = 127, 2
  t = trange(start_seed, start_seed + n_iters, desc='Run #', leave=True)
  seeds = list(range(start_seed, start_seed + n_iters))

  workdir = Path(f'data/workdir/{model_name}')
  vectordir = workdir/'vectordir'
  modeldir = workdir/'models'
  targs.append(y_test)

  for seed in t:
    t.set_description(f"Run # (seed {seed})")
    df = set_group_splits(data_df.copy(), group_col='hadm_id', seed=seed, pct=0.15)

    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    y_train, y_test = train_df['imi_adm_label'], test_df['imi_adm_label']

    if subset == 's':
      x_train, x_test = train_df[str_cols].values, test_df[str_cols].values
    elif subest == 'u':
      with open(vectordir/'bigram_{seed}.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        x_train = pickle.load(f)
        x_test = pickle.load(f)
    elif subset == 'u+s':
      x_vitals_train, x_vitals_test = train_df[str_cols].values, test_df[str_cols].values
      with open(vectordir/'bigram_{seed}.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        x_note_train = pickle.load(f)
        x_note_test = pickle.load(f)      

      x_train = scipy.sparse.hstack((x_vitals_train, x_note_train)).tocsr()
      x_test = scipy.sparse.hstack((x_vitals_test, x_note_test)).tocsr()  

    clf = clf_model(**clf_params)
    clf.fit(x_train, y_train)
    clfs.append(clf)

    prob = clf.predict_proba(x_test)
    probs.append(prob)

    y_pred = (prob[:, 1] > threshold).astype(np.int64)
    preds.append(y_pred)


  for i, clf in enumerate(clfs):
    pickle.dump(clf, open(modeldir/f'{subset}_seed_{start_seed + i}.pkl', 'wb'))

  with open(workdir/f'{subset}_preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(probs, f)
    pickle.dump(preds, f)
