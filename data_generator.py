import pandas as pd
import numpy as np
import json
from datetime import datetime

#------- generating ICU Stay ID files ---------#
sepsis3_df = pd.read_csv('./data/sepsis3-df.csv')

icustay_ids = sepsis3_df['icustay_id']
icustay_ids.to_csv('./data/icustay_ids.txt', index=False)

icu2hadm_df = sepsis3_df[['icustay_id', 'hadm_id']]
icu2hadm_dict = icu2hadm_df.set_index('icustay_id').to_dict()['hadm_id']
icu2hadm_json = json.dumps(icu2hadm_dict)
dict_file = open('./data/icu_hadm_dict.json', "w")
dict_file.write(icu2hadm_json)
dict_file.close()

#------- generating treatment/{treatment_option}/{ID}.npy files ------#
vaso_df = pd.read_csv('./data/vaso_durations.csv')
vent_df = pd.read_csv('./data/vent_durations.csv')

# hadm ids for sepsis patient in MIMIC-III
icu2hadm = pd.read_json('./data/icu_hadm_dict.json', typ='series').to_dict()
icu_id_sepsis = list(icu2hadm.keys())
hadm_id_sepsis = list(icu2hadm.values())

# prune to sepsis3 cohort only
vaso_df = vaso_df[vaso_df['icustay_id'].isin(icu_id_sepsis)]
vent_df = vent_df[vent_df['icustay_id'].isin(icu_id_sepsis)]

# map icustay_id to hadm_id
vaso_df = vaso_df.replace({'icustay_id': icu2hadm})
vaso_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
vent_df = vent_df.replace({'icustay_id': icu2hadm})
vent_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)

# save each patient demographic details to static.npy file
for treatment_option, df in [('vaso', vaso_df), ('vent', vent_df)]:
  for ID in list(vaso_df['hadm_id']):
    patient = df[df['hadm_id'] == ID]
    patient = patient['duration_hours'].to_numpy()
    # adjust length of arr st it matches the expected A format
    if len(patient) < 10:
      zeros = np.zeros(10-len(patient))
      patient = np.append(patient, zeros)
    elif len(patient) > 10:
      patient = patient[:10]
    patient = patient.reshape((10, 1))
    np.save('./data/treatment/{}/{}.npy'.format(treatment_option, ID), patient)

#------- generating /static/{ID}.static.npy files ------#
detail_df = pd.read_csv('./data/detail.csv')
comorbid_df = pd.read_csv('./data/comorbid.csv')
height_weight_df = pd.read_csv('./data/height_weight.csv')

# prune to sepsis3 cohort only
detail_df = detail_df[detail_df['hadm_id'].isin(hadm_id_sepsis)]
comorbid_df = comorbid_df[comorbid_df['hadm_id'].isin(hadm_id_sepsis)]
height_weight_df = height_weight_df[height_weight_df['icustay_id'].isin(icu_id_sepsis)]

#change height_weight to use hadm_id
height_weight_df = height_weight_df.replace({'icustay_id': icu2hadm})
height_weight_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)

# calculate race variable
detail_df['white'] = detail_df['ethnicity_grouped']=='white'
detail_df['black'] = detail_df['ethnicity_grouped']=='black'
detail_df['hispanic'] = detail_df['ethnicity_grouped']=='hispanic'
detail_df['asian'] = detail_df['ethnicity_grouped']=='asian'
detail_df['other'] = ~(detail_df['white'] | detail_df['black'] | detail_df['hispanic'] | detail_df['asian'])
detail_df.drop(columns=['ethnicity_grouped'], inplace=True)

# calculate bmi variable
height_weight_df['bmi'] = height_weight_df['weight_first'] / (height_weight_df['height_first']/100)**2

# merging all time varying data into one df
static_df = detail_df.merge(comorbid_df, on='hadm_id', how='inner')
static_df = static_df.merge(height_weight_df, on='hadm_id', how='inner')

# save each patient demographic details to static.npy file
for ID in list(static_df['hadm_id']):
  patient = static_df[static_df['hadm_id'] == ID]
  patient = patient.loc[:, patient.columns!='hadm_id']
  np.save('./data/static/{}.static.npy'.format(ID), patient.to_numpy()[0])

# # ------- generating time variable x/{ID}.csv files ------#
# vitals_df = pd.read_csv('./data/pivoted_vitals.csv')
# labs_df = pd.read_csv('./data/pivoted_labs.csv')
# gcs_df = pd.read_csv('./data/pivoted_gcs.csv')
# uo_df = pd.read_csv('./data/urine_output.csv')

# # keep only sepsis cohort for all data (vitals, labs, gcs, uo)
# vitals_df = vitals_df[vitals_df['icustay_id'].isin(icu_id_sepsis)]
# labs_df = labs_df[labs_df['hadm_id'].isin(hadm_id_sepsis)]
# gcs_df = gcs_df[gcs_df['icustay_id'].isin(icu_id_sepsis)]
# uo_df = uo_df[uo_df['icustay_id'].isin(icu_id_sepsis)]

# # change icustay_id to corresponding hadm_id in vitals, gcs, uo
# vitals_df = vitals_df.replace({'icustay_id': icu2hadm})
# gcs_df = gcs_df.replace({'icustay_id': icu2hadm})
# uo_df = uo_df.replace({'icustay_id': icu2hadm})

# # change col name from icustay_id to hadm_id
# vitals_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
# gcs_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
# uo_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)

# # merging all time varying data into one df
# all_labs_df = labs_df.merge(gcs_df, on='hadm_id', how='left')
# all_labs_df = all_labs_df.merge(uo_df, on='hadm_id', how='left')
# time_var_df = vitals_df.merge(all_labs_df, on='hadm_id', how='inner')
# var_rename = {'charttime':'time','HEMOGLOBIN':'hemoglobin','HeartRate':'heartrate','CREATININE':'creatinine','HEMATOCRIT':'hematocrit','SysBP':'sysbp','TempC':'tempc','PT':'pt','SODIUM':'sodium','DiasBP':'diasbp', 'GCS':'gcs_min','PLATELET':'platelet','PTT':'ptt','CHLORIDE':'chloride','RespRate':'resprate','GLUCOSE':'glucose','BICARBONATE':'bicarbonate','BANDS':'bands', 'BUN':'bun','value':'urineoutput','INR':'inr','LACTATE':'lactate','ANIONGAP':'aniongap','SpO2':'spo2','WBC':'wbc','MeanBP':'meanbp'}
# time_var_df.rename(columns=var_rename, inplace=True)

# # saving each patient data to a file
# for ID in list(time_var_df['hadm_id']):
#   patient = time_var_df[time_var_df['hadm_id'] == ID]
#   patient = patient.loc[:, patient.columns!='hadm_id']
#   # prune to only first 30 hours of a patient's stay
#   times_str = set(patient['time'])
#   times = []
#   for time_str in times_str:
#     times.append(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'))
#   times = sorted(times)
#   times = times[1:32:3]
#   sorted_str_times = []
#   for time in times:
#     sorted_str_times.append(time.strftime('%Y-%m-%d %H:%M:%S'))
#   times_dict = dict(zip(sorted_str_times, list(range(1, 32, 3))))
#   patient = patient[patient['time'].isin(sorted_str_times)]
#   patient = patient.replace({'time':times_dict})
#   # fill NULL values with mean of column
#   means = patient.mean(axis=0)
#   for col in patient.columns:
#     patient[col] = patient[col].fillna(means[col])
#   patient.to_csv('./data/x/{}.csv'.format(ID), index=False)