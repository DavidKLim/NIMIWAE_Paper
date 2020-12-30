import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

def pivot_dataframe(df):
    assert len(df) > 6, "DataFrame does not contain any time series information"
    df_dropped = df.drop([0,1,2,3,4,5])
    df_dropped.Time = df_dropped.Time.apply(lambda t: int(t.split(":")[0]))
    df_pivoted = df_dropped.pivot_table(index="Time", columns="Parameter", values="Value")
    column_list = ['ALP','ALT','AST','Albumin','BUN','Bilirubin','Cholesterol','Creatinine','DiasABP','FiO2','GCS',
               'Glucose','HCO3','HCT','HR','K','Lactate','MAP','MechVent','Mg','NIDiasABP','NIMAP','NISysABP','Na',
               'PaCO2','PaO2','Platelets','RespRate','SaO2','SysABP','Temp','TroponinI','TroponinT','Urine','WBC','pH']
    df_reindexed = df_pivoted.reindex(column_list, axis=1)
    df_reindexed2 = df_reindexed.reindex(range(48), axis=0)
    return df_reindexed2

train_files = glob("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-a/*.txt")
df = pivot_dataframe(pd.read_csv(train_files[0]))

for df_file in tqdm(train_files):
    patient_id = os.path.basename(df_file).split(".")[0]
    try:
        df = pivot_dataframe(pd.read_csv(df_file))
        print(df.head())
        df.to_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_a_merged.h5', "id_"+patient_id, complib='zlib', complevel=5)
    except:
        continue

val_files = glob("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-b/*.txt")
test_files = glob("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set-c/*.txt")

for df_file in tqdm(val_files):
    patient_id = os.path.basename(df_file).split(".")[0]
    try:
        df = pivot_dataframe(pd.read_csv(df_file))
        print(df.head())
        df.to_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_b_merged.h5', "id_"+patient_id, complib='zlib', complevel=5)
    except:
        continue

for df_file in tqdm(test_files):
    patient_id = os.path.basename(df_file).split(".")[0]
    try:
        df = pivot_dataframe(pd.read_csv(df_file))
        print(df.head())
        df.to_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_c_merged.h5', "id_"+patient_id, complib='zlib', complevel=5)
    except:
        continue

with pd.HDFStore('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_a_merged.h5') as hdf:
    hdf_keys = hdf.keys()

x_train = np.zeros([len(hdf_keys), 48, 36])
m_train = np.zeros([len(hdf_keys), 48, 36])

for i, key in tqdm(enumerate(hdf_keys), total=len(hdf_keys)):
    df = pd.read_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_a_merged.h5', key)
    x_train[i] = df.fillna(0.).to_numpy()
    df_miss = df.copy()
    df_miss[df_miss.notnull()] = 0.
    m_train[i] = df_miss.fillna(1.).to_numpy()

with pd.HDFStore('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_b_merged.h5') as hdf:
    hdf_keys = hdf.keys()
    
x_val = np.zeros([len(hdf_keys), 48, 36])
m_val = np.zeros([len(hdf_keys), 48, 36])

for i, key in tqdm(enumerate(hdf_keys), total=len(hdf_keys)):
    df = pd.read_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_b_merged.h5', key)
    x_val[i] = df.fillna(0.).to_numpy()
    df_miss = df.copy()
    df_miss[df_miss.notnull()] = 0.
    m_val[i] = df_miss.fillna(1.).to_numpy()
    
with pd.HDFStore('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_c_merged.h5') as hdf:
    hdf_keys = hdf.keys()
    
x_test = np.zeros([len(hdf_keys), 48, 36])
m_test = np.zeros([len(hdf_keys), 48, 36])

for i, key in tqdm(enumerate(hdf_keys), total=len(hdf_keys)):
    df = pd.read_hdf('predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_c_merged.h5', key)
    x_test[i] = df.fillna(0.).to_numpy()
    df_miss = df.copy()
    df_miss[df_miss.notnull()] = 0.
    m_test[i] = df_miss.fillna(1.).to_numpy()
    
np.savez_compressed("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/data_train_val.npz", x_train=x_train, m_train=m_train, x_val=x_val, m_val=m_val)
np.savez_compressed("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/data_test.npz", x_test=x_test, m_test=m_test)
    


