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
import lightgbm as lgb

from utils.data_utils import set_two_splits
from args import args

def process_str(train_df, valid_df):
  brands = train_df.groupby('brand_name')['price'].mean().sort_values(ascending=False).to_frame()
  brands['id'] = brands.reset_index().index.values
  brand_names = brands.index.values

  train_brand_data = brands.loc[train_df['brand_name']]
  train_df.loc[:, 'brand_val'] = train_brand_data['id'].values/len(brand_names)

  valid_brand_data = brands.loc[valid_df['brand_name']]
  valid_df.loc[:, 'brand_val'] = valid_brand_data['id'].values/len(brand_names)

  maincats = train_df.groupby('main_cat')['price'].mean().sort_values(ascending=False).to_frame()
  maincats['id'] = maincats.reset_index().index.values
  maincat_names = maincats.index.values

  train_maincats = maincats.loc[train_df['main_cat']]
  train_df.loc[:, 'maincat_val'] = train_maincats['id'].values/len(maincat_names)

  valid_maincats = maincats.loc[valid_df['main_cat']]
  valid_df.loc[:, 'maincat_val'] = valid_maincats['id'].values/len(maincat_names)

  subcat1s = train_df.groupby('sub_cat1')['price'].mean().sort_values(ascending=False).to_frame()
  subcat1s['id'] = subcat1s.reset_index().index.values
  subcat1_names = subcat1s.index.values

  train_subcat1s = subcat1s.loc[train_df['sub_cat1']]
  train_df.loc[:, 'subcat1_val'] = train_subcat1s['id'].values/len(subcat1_names)

  valid_subcat1s = subcat1s.loc[valid_df['sub_cat1']]
  valid_df.loc[:, 'subcat1_val'] = valid_subcat1s['id'].values/len(subcat1_names)

  subcat2s = train_df.groupby('sub_cat2')['price'].mean().sort_values(ascending=False).to_frame()
  subcat2s['id'] = subcat2s.reset_index().index.values
  subcat2_names = subcat2s.index.values

  train_subcat2s = subcat2s.loc[train_df['sub_cat2']]
  train_df.loc[:, 'subcat2_val'] = train_subcat2s['id'].values/len(subcat2_names)

  valid_subcat2s = subcat2s.loc[valid_df['sub_cat2']]
  valid_df.loc[:, 'subcat2_val'] = valid_subcat2s['id'].values/len(subcat2_names)  

  x_train = train_df[['item_condition_id', 'shipping', 'brand_val', 'maincat_val', 'subcat1_val', 'subcat2_val']].values
  x_valid = valid_df[['item_condition_id', 'shipping', 'brand_val', 'maincat_val', 'subcat1_val', 'subcat2_val']].values  

  return x_train, x_valid

def process_ustr(train_df, valid_df):
  x_train = vectorizer.fit_transform(train_df['text'].values.astype('U'))
  x_valid = vectorizer.transform(valid_df['text'].values.astype('U'))

  return x_train, x_valid

if __name__=='__main__':
  if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} subset (s|u|u+s)")
    sys.exit(1)

  subset = sys.argv[1]

  str_cols = ['item_condition_id', 'brand_name', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']
  data_df = pd.read_csv(args.path/'train_df.csv')

  clf_params = {
    'num_leaves': 400,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'metric': 'rmse',
    'num_threads': 32,
    'max_bin': 32,
    'objective': 'regression',
  }

  clfs, targs, preds = [], [], []
  start_seed, n_iters = 127, 1
  t = trange(start_seed, start_seed + n_iters, desc='Run #', leave=True)

  for seed in t:
    t.set_description(f"Run # (seed {seed})")
    df = set_two_splits(data_df.copy(),  name='valid', seed=seed, pct=0.15)

    if subset == 's':
      train_df = df.loc[df['split'] == 'train', str_cols+['price']].reset_index(drop=True)
      valid_df = df.loc[df['split'] == 'valid', str_cols+['price']].reset_index(drop=True)
      x_train, x_valid = process_str(train_df, valid_df)
    elif subset == 'u':
      train_df = df.loc[df['split'] == 'train', ['text', 'price']].reset_index(drop=True)
      valid_df = df.loc[df['split'] == 'valid', ['text, ''price']].reset_index(drop=True)
      x_train, x_valid = process_ustr(train_df, valid_df)
    elif subset == 'u+s':
      train_df = df.loc[df['split'] == 'train'].reset_index(drop=True)
      valid_df = df.loc[df['split'] == 'valid'].reset_index(drop=True)
      x_str_train, x_str_valid = process_str(train_df, valid_df)
      x_ustr_train, x_ustr_valid = process_ustr(train_df, valid_df)

      x_train = scipy.sparse.hstack((x_str_train, x_ustr_train)).tocsr()
      x_valid = scipy.sparse.hstack((x_str_valid, x_ustr_valid)).tocsr()

    y_train, y_valid = train_df['price'], valid_df['price']
    print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)


    # with open(args.workdir/f'full_common/vectordir/{vec_prefix}_{seed}.pkl', 'rb') as f:
    #   vectorizer = pickle.load(f)
    #   x_note_train = pickle.load(f)
    #   x_note_test = pickle.load(f)

    # y_train, y_test = train_df['imi_adm_label'], test_df['imi_adm_label']
    # targs.append(y_test)

    # x_str_train, x_str_test = train_df[args.str_cols].values, test_df[args.str_cols].values
    # x_train = scipy.sparse.hstack((x_str_train, x_note_train)).tocsr()
    # x_test = scipy.sparse.hstack((x_str_test, x_note_test)).tocsr()

    # clf = clf_model(**clf_params)
    # clf.fit(x_train, y_train)
    # clfs.append(clf)

    # prob = clf.predict_proba(x_test)
    # probs.append(prob)

    # y_pred = (prob[:, 1] > threshold).astype(np.int64)
    # preds.append(y_pred)

      

  # model_name = sys.argv[1]
  # cohort = sys.argv[2]

  # if model_name == 'lr':
  #   args = lr_args
  #   clf_model = LogisticRegression
  #   clf_params = {
  #     'class_weight': 'balanced',
  #   }
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
  # else:
  #   print("Allowed models: (lr|rf|gbm)")
  #   sys.exit(1)

  # if cohort == 'vital':
  #   threshold = args.full_common_vital_threshold
  #   vec_prefix = 'bi_gram_vital'
  # elif cohort == 'all':
  #   threshold = args.full_common_all_threshold
  #   vec_prefix = 'bi_gram_all'
  # else:
  #   print("Allowed cohort: (vital|all)")
  #   sys.exit(1)


  # args.str_cols = pickle.load(open(f'data/str_{cohort}_cols.pkl', 'rb'))
  # args.cols=['hadm_id'] + args.str_cols + ['note', 'imi_adm_label']

  # data_df = pd.read_csv(f'data/full_common_{cohort}.csv', usecols=args.cols)
  # data_df = data_df[data_df['imi_adm_label'] != -1].reset_index(drop=True)

  # clfs, targs, preds, probs = [], [], [], []
  # start_seed, n_iters = 127, 100
  # t = trange(start_seed, start_seed + n_iters, desc='Run #', leave=True)
  # # seeds = list(range(start_seed, start_seed + n_iters))

  # args.workdir = Path('data/workdir')
  # args.modeldir = args.workdir/f'full_common/{model_name}/models'

  # for seed in t:
  #   t.set_description(f"Run # (seed {seed})")
  #   df = set_group_splits(data_df.copy(), group_col='hadm_id', seed=seed, pct=0.15)
  #   train_df = df[df['split'] == 'train']
  #   test_df = df[df['split'] == 'test']

  #   with open(args.workdir/f'full_common/vectordir/{vec_prefix}_{seed}.pkl', 'rb') as f:
  #     vectorizer = pickle.load(f)
  #     x_note_train = pickle.load(f)
  #     x_note_test = pickle.load(f)

  #   y_train, y_test = train_df['imi_adm_label'], test_df['imi_adm_label']
  #   targs.append(y_test)

  #   x_str_train, x_str_test = train_df[args.str_cols].values, test_df[args.str_cols].values
  #   x_train = scipy.sparse.hstack((x_str_train, x_note_train)).tocsr()
  #   x_test = scipy.sparse.hstack((x_str_test, x_note_test)).tocsr()

  #   clf = clf_model(**clf_params)
  #   clf.fit(x_train, y_train)
  #   clfs.append(clf)

  #   prob = clf.predict_proba(x_test)
  #   probs.append(prob)

  #   y_pred = (prob[:, 1] > threshold).astype(np.int64)
  #   preds.append(y_pred)


  # for i, clf in enumerate(clfs):
  #   pickle.dump(clf, open(args.modeldir/f'full_common_{cohort}_seed_{start_seed + i}.pkl', 'wb'))

  # with open(args.workdir/f'full_common/{model_name}/full_common_{cohort}_preds.pkl', 'wb') as f:
  #   pickle.dump(targs, f)
  #   pickle.dump(probs, f)
  #   pickle.dump(preds, f)
