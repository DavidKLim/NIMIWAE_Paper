# https://github.com/alistairewj/challenge2012/blob/master/prepare-data.ipynb

import pandas as pd
import numpy as np
import os

# pick a set
# dataset = 'set-a'
# dataset = 'set-b'
# dataset = 'set-c'
def process_Alistair(dataset):
  # load all files into list of lists
  txt_all = list()
  for f in os.listdir(dataset):
      with open(os.path.join(dataset, f), 'r') as fp:
          txt = fp.readlines()
          
      # get recordid to add as a column
      recordid = txt[1].rstrip('\n').split(',')[-1]
      txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
      txt_all.extend(txt[1:])
  
  
  # convert to pandas dataframe
  df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])
  
  # extract static variables into a separate dataframe
  df_static = df.loc[df['time'] == '00:00', :].copy()
  
  # retain only one of the 6 static vars:
  static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
  df_static = df_static.loc[df['parameter'].isin(static_vars)]
  
  # remove these from original df
  idxDrop = df_static.index
  df = df.loc[~df.index.isin(idxDrop), :]
  
  # to ensure there are no duplicates, group by recordid/parameter and take the last value
  # last will be chosen as last row in the loaded file
  # there was 1 row in set-b which had 2 weights (70.4, 70.8) and thus required this step
  df_static = df_static.groupby(['recordid', 'parameter'])[['value']].last()
  df_static.reset_index(inplace=True)
  
  # pivot on parameter so there is one column per parameter
  df_static = df_static.pivot(index='recordid', columns='parameter', values='value')
  
  # some conversions on columns for convenience
  df['value'] = pd.to_numeric(df['value'], errors='raise')
  df['time'] = df['time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
  
  df.head()
  
  features = {'Albumin': 'Serum Albumin (g/dL)',
      'ALP': 'Alkaline phosphatase (IU/L)',
      'ALT': 'Alanine transaminase (IU/L)',
      'AST': 'Aspartate transaminase (IU/L)',
      'Bilirubin': 'Bilirubin (mg/dL)',
      'BUN': 'Blood urea nitrogen (mg/dL)',
      'Cholesterol': 'Cholesterol (mg/dL)',
      'Creatinine': 'Serum creatinine (mg/dL)',
      'DiasABP': 'Invasive diastolic arterial blood pressure (mmHg)',
      'FiO2': 'Fractional inspired O2 (0-1)',
      'GCS': 'Glasgow Coma Score (3-15)',
      'Glucose': 'Serum glucose (mg/dL)',
      'HCO3': 'Serum bicarbonate (mmol/L)',
      'HCT': 'Hematocrit (%)',
      'HR': 'Heart rate (bpm)',
      'K': 'Serum potassium (mEq/L)',
      'Lactate': 'Lactate (mmol/L)',
      'Mg': 'Serum magnesium (mmol/L)',
      'MAP': 'Invasive mean arterial blood pressure (mmHg)',
      'MechVent': 'Mechanical ventilation respiration (0:false or 1:true)',
      'Na': 'Serum sodium (mEq/L)',
      'NIDiasABP': 'Non-invasive diastolic arterial blood pressure (mmHg)',
      'NIMAP': 'Non-invasive mean arterial blood pressure (mmHg)',
      'NISysABP': 'Non-invasive systolic arterial blood pressure (mmHg)',
      'PaCO2': 'partial pressure of arterial CO2 (mmHg)',
      'PaO2': 'Partial pressure of arterial O2 (mmHg)',
      'pH': 'Arterial pH (0-14)',
      'Platelets': 'Platelets (cells/nL)',
      'RespRate': 'Respiration rate (bpm)',
      'SaO2': 'O2 saturation in hemoglobin (%)',
      'SysABP': 'Invasive systolic arterial blood pressure (mmHg)',
      'Temp': 'Temperature (°C)',
      'TroponinI': 'Troponin-I (μg/L)',
      'TroponinT': 'Troponin-T (μg/L)',
      'Urine': 'Urine output (mL)',
      'WBC': 'White blood cell count (cells/nL)',
      'Weight': 'Weight (kg)'}
  
  # convert static into numeric
  for c in df_static.columns:
      df_static[c] = pd.to_numeric(df_static[c])
      
  # preprocess
  for c in df_static.columns:
      x = df_static[c]
      if c == 'Age':
          # replace anon ages with 91.4
          idx = x > 130
          df_static.loc[idx, c] = 91.4
      elif c == 'Gender':
          idx = x < 0
          df_static.loc[idx, c] = np.nan
      elif c == 'Height':
          idx = x < 0
          df_static.loc[idx, c] = np.nan
          
          # fix incorrectly recorded heights
          
          # 1.8 -> 180
          idx = x < 10
          df_static.loc[idx, c] = df_static.loc[idx, c] * 100
          
          # 18 -> 180
          idx = x < 25
          df_static.loc[idx, c] = df_static.loc[idx, c] * 10
          
          # 81.8 -> 180 (inch -> cm)
          idx = x < 100
          df_static.loc[idx, c] = df_static.loc[idx, c] * 2.2
          
          # 1800 -> 180
          idx = x > 1000
          df_static.loc[idx, c] = df_static.loc[idx, c] * 0.1
          
          # 400 -> 157
          idx = x > 250
          df_static.loc[idx, c] = df_static.loc[idx, c] * 0.3937
          
      elif c == 'Weight':
          idx = x < 35
          df_static.loc[idx, c] = np.nan
          
          idx = x > 299
          df_static.loc[idx, c] = np.nan
  
  def delete_value(df, c, value=0):
      idx = df['parameter'] == c
      idx = idx & (df['value'] == value)
      
      df.loc[idx, 'value'] = np.nan
      return df
  
  def replace_value(df, c, value=np.nan, below=None, above=None):
      idx = df['parameter'] == c
      
      if below is not None:
          idx = idx & (df['value'] < below)
          
      if above is not None:
          idx = idx & (df['value'] > above)
      
      
      if 'function' in str(type(value)):
          # value replacement is a function of the input
          df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(value)
      else:
          df.loc[idx, 'value'] = value
          
      return df
  
  
  # delete/replace outliers/nonsensical values
  df = delete_value(df, 'DiasABP', -1)
  df = replace_value(df, 'DiasABP', value=np.nan, below=1)
  df = replace_value(df, 'DiasABP', value=np.nan, above=200)
  df = replace_value(df, 'SysABP', value=np.nan, below=1)
  df = replace_value(df, 'MAP', value=np.nan, below=1)
  
  df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
  df = replace_value(df, 'NISysABP', value=np.nan, below=1)
  df = replace_value(df, 'NIMAP', value=np.nan, below=1)
  
  df = replace_value(df, 'HR', value=np.nan, below=1)
  df = replace_value(df, 'HR', value=np.nan, above=299)
  
  df = replace_value(df, 'PaCO2', value=np.nan, below=1)
  df = replace_value(df, 'PaCO2', value=lambda x: x*10, below=10)
  
  df = replace_value(df, 'PaO2', value=np.nan, below=1)
  df = replace_value(df, 'PaO2', value=lambda x: x*10, below=20)
  
  # the order of these steps matters
  df = replace_value(df, 'pH', value=lambda x: x*10, below=0.8, above=0.65)
  df = replace_value(df, 'pH', value=lambda x: x*0.1, below=80, above=65)
  df = replace_value(df, 'pH', value=lambda x: x*0.01, below=800, above=650)
  df = replace_value(df, 'pH', value=np.nan, below=6.5)
  df = replace_value(df, 'pH', value=np.nan, above=8.0)
  
  # convert to farenheit
  df = replace_value(df, 'Temp', value=lambda x: x*9/5+32, below=10, above=1)
  df = replace_value(df, 'Temp', value=lambda x: (x-32)*5/9, below=113, above=95)
  
  df = replace_value(df, 'Temp', value=np.nan, below=25)
  df = replace_value(df, 'Temp', value=np.nan, above=45)
  
  df = replace_value(df, 'RespRate', value=np.nan, below=1)
  df = replace_value(df, 'WBC', value=np.nan, below=1)
  
  df = replace_value(df, 'Weight', value=np.nan, below=35)
  df = replace_value(df, 'Weight', value=np.nan, above=299)
  
  
  # Initialize a dataframe with df_static
  X = df_static.copy()
  
  X.drop('RecordID', axis=1, inplace=True)
  
  # MICU is ICUType==3, and is used as the reference category
  X['CCU'] = (X['ICUType'] == 1).astype(int)
  X['CSRU'] = (X['ICUType'] == 2).astype(int)
  X['SICU'] = (X['ICUType'] == 4).astype(int)
  X.drop('ICUType', axis=1, inplace=True)
  
  # For the following features we extract: first, last, lowest, highest, median
  feats = ['DiasABP', 'GCS', 'Glucose', 'HR', 'MAP',
  'NIDiasABP', 'NIMAP', 'NISysABP', 
  'RespRate', 'SaO2', 'Temp', ]
  
  idx = df['parameter'].isin(feats)
  df_tmp = df.loc[idx, :].copy()
  df_tmp = df_tmp.groupby(['recordid', 'parameter'])['value']
  
  for agg in ['first', 'last', 'lowest', 'highest', 'median']:
      if agg == 'first':
          X_add = df_tmp.first()
      elif agg == 'last':
          X_add = df_tmp.last()
      elif agg == 'lowest':
          X_add = df_tmp.min()
      elif agg == 'highest':
          X_add = df_tmp.max()
      elif agg == 'median':
          X_add = df_tmp.median()
      else:
          print('Unrecognized aggregation {}. Skipping.'.format(agg))
          
      X_add = X_add.reset_index()
      X_add = X_add.pivot(index='recordid', columns='parameter', values='value')
      X_add.columns = [x + '_' + agg for x in X_add.columns]
  
      X = X.merge(X_add, how='left', left_index=True, right_index=True)
  
  
  # For the following features we extract: first, last
  feats = ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
  'Creatinine', 'FiO2', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na',
  'PaCO2', 'PaO2', 'pH', 'Platelets', 'SysABP', 'TroponinI', 'TroponinT',
  'WBC', 'Weight']
  
  
  idx = df['parameter'].isin(feats)
  df_tmp = df.loc[idx, :].copy()
  df_tmp = df_tmp.groupby(['recordid', 'parameter'])['value']
  
  for agg in ['first', 'last']:
      if agg == 'first':
          X_add = df_tmp.first()
      elif agg == 'last':
          X_add = df_tmp.last()
      elif agg == 'lowest':
          X_add = df_tmp.min()
      elif agg == 'highest':
          X_add = df_tmp.max()
      elif agg == 'median':
          X_add = df_tmp.median()
      else:
          print('Unrecognized aggregation {}. Skipping.'.format(agg))
          
      X_add = X_add.reset_index()
      X_add = X_add.pivot(index='recordid', columns='parameter', values='value')
      X_add.columns = [x + '_' + agg for x in X_add.columns]
  
      X = X.merge(X_add, how='left', left_index=True, right_index=True)
  
  # For the following features we extract custom data
  idx = df['parameter'] == 'MechVent'
  df_tmp = df.loc[idx, :].copy().groupby('recordid')
  
  X0 = df_tmp[['time']].min()
  X0.columns = ['MechVentStartTime']
  
  X1 = df_tmp[['time']].max()
  X1.columns = ['MechVentEndTime']
  
  X_add = X0.merge(X1, how='inner',
                   left_index=True, right_index=True)
  X_add['MechVentDuration'] = X_add['MechVentEndTime'] - X_add['MechVentStartTime']
  
  X_add['MechVentLast8Hour'] = (X_add['MechVentEndTime'] >= 2400).astype(int)
  X_add.drop('MechVentEndTime', axis=1, inplace=True)
  
  X = X.merge(X_add, how='left', left_index=True, right_index=True)
  
  # Urine output
  idx = df['parameter'] == 'MechVent'
  df_tmp = df.loc[idx, :].copy().groupby('recordid')
  
  X_add = df_tmp[['value']].sum()
  X_add.columns = ['UrineOutputSum']
  
  X = X.merge(X_add, how='left', left_index=True, right_index=True)
  
  print(X.shape)
  X.head()
  
  
  # load in outcomes
  if dataset == 'set-a':
      y = pd.read_csv('Outcomes-a.txt')
  elif dataset == 'set-b':
      y = pd.read_csv('Outcomes-b.txt')
  elif dataset == 'set-c':
      y = pd.read_csv('Outcomes-c.txt')
      
  y.set_index('RecordID', inplace=True)
  y.index.name = 'recordid'
  X = y.merge(X, how='inner', left_index=True, right_index=True)
  X.head()
  
  
  # output to file
  X.to_csv('PhysionetChallenge2012-{}.csv'.format(dataset),
           sep=',', index=True)
