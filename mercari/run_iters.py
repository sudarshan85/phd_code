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
from sklearn.metrics import mean_squared_error
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

def process_ustr(seed):
  with open(args.vectordir/f'bigram_{seed}.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
    x_train = pickle.load(f)
    x_valid = pickle.load(f)

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
    'verbose': -1
  }

  clfs, targs, preds = [], [], []
  start_seed, n_iters = 127, 10
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
      valid_df = df.loc[df['split'] == 'valid', ['text', 'price']].reset_index(drop=True)
      x_train, x_valid = process_ustr(seed)
    elif subset == 'u+s':
      train_df = df.loc[df['split'] == 'train'].reset_index(drop=True)
      valid_df = df.loc[df['split'] == 'valid'].reset_index(drop=True)
      x_str_train, x_str_valid = process_str(train_df, valid_df)
      x_ustr_train, x_ustr_valid = process_ustr(seed)

      x_train = scipy.sparse.hstack((x_str_train, x_ustr_train)).tocsr()
      x_valid = scipy.sparse.hstack((x_str_valid, x_ustr_valid)).tocsr()
    else:
      print(f"Unknown subset {subset}")
      sys.exit(1)

    y_train, y_valid = train_df['price'], valid_df['price']
    targs.append(y_valid)

    train_ds = lgb.Dataset(x_train, y_train)
    valid_ds = lgb.Dataset(x_valid, y_valid, reference=train_ds)

    gbm = lgb.train(clf_params, train_ds, num_boost_round=600, valid_sets=[train_ds, valid_ds], early_stopping_rounds=10, verbose_eval=True)
    clfs.append(gbm)

    y_pred = gbm.predict(x_valid)
    preds.append(y_pred)
    print(np.round(np.sqrt(mean_squared_error(y_valid, y_pred)), 3))

  for i, clf in enumerate(clfs):
    pickle.dump(clf, open(args.modeldir/f'clf_seed_{start_seed + i}.pkl', 'wb'))

  with open(args.workdir/f'{subset}_preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(preds, f)
