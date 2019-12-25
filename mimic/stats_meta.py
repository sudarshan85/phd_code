#!/usr/bin/env python

import sys
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from scipy import stats
from ast import literal_eval

path = Path('data')
stats_dir = path/'statsdir'

def change_name(col_name):
  if '(' not in col_name:
    return col_name
  cols = literal_eval(col_name)
  return f'{cols[0]}_{cols[1]}'

if __name__=='__main__':
  print("Loading file")
  vitals_meta_common = pd.read_csv('data/structured_vitals_proc_with_meta.csv', parse_dates=['intime', 'admittime', 'charttime'])
  running_stats = ['min', 'mean', 'median', 'std', 'max']
  var_cols = vitals_meta_common.columns[4:]
  dfs = []

  # hadms = [134899, 137495, 161246, 171847, 187987]
  # subset_df = vitals_meta_common.loc[(vitals_meta_common['hadm_id'].isin(hadms))].reset_index(drop=True)
  # for hadm_id, group_df in tqdm(subset_df.groupby('hadm_id'), desc='Encounters'):
  for hadm_id, group_df in tqdm(vitals_meta_common.groupby('hadm_id'), desc='Encounters'):
    df = group_df.copy()
    intime = df['intime']
    admittime = df['admittime']
    var_df = df[var_cols].reset_index(drop=True) # save the original vals for later
    
    df.set_index('charttime', inplace=True) # set charttime as index for rolling 24h
    stats_df = df[var_cols].rolling('24h').agg(running_stats)
    
    df = pd.DataFrame(stats_df.to_records()) # flatten the resulting dataframe
    df.insert(loc=1, column='hadm_id', value=hadm_id)
    df.insert(loc=1, column='intime', value=intime)
    df.insert(loc=1, column='admittime', value=admittime)
    
    df.rename(columns=change_name, inplace=True) # rename columns
    df = pd.concat([df, var_df], axis=1) # add the original vals back
    
    # reorder vars such that the columns are var, var_stat...
    stats_meta_cols = df.columns[4:]
    all_cols = []
    for var in var_cols:
      all_cols.append(var)
      for stat in stats_meta_cols:
        if f'{var}_' in stat:
          all_cols.append(stat)
          
    order = list(df.columns[:4]) + all_cols
    df = df[order]  
    dfs.append(df)

  vitals_meta_common_stats = pd.concat(dfs)
  vitals_meta_common_stats.reset_index(drop=True, inplace=True)
  vitals_meta_common_stats['charttime'] = pd.to_datetime(vitals_meta_common_stats['charttime'])
  vitals_meta_common_stats['intime'] = pd.to_datetime(vitals_meta_common_stats['intime'])
  vitals_meta_common_stats['admittime'] = pd.to_datetime(vitals_meta_common_stats['admittime'])  

  # fill first occurance of std which is nan with 0
  std_cols = [col for col in vitals_meta_common_stats.columns if 'std' in col]
  vitals_meta_common_stats[std_cols] = vitals_meta_common_stats[std_cols].fillna(0)

  cols = ['hadm_id', 'admittime', 'intime', 'charttime'] + list(vitals_meta_common_stats.columns[2:])
  vitals_meta_common_stats = vitals_meta_common_stats[cols]

  vitals_meta_common_stats.to_csv('data/structured_vitals_stats_with_meta.csv', index=False)