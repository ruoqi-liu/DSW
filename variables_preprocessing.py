import pandas as pd

# hadm ids for sepsis patient in MIMIC-III
icu2hadm = pd.read_json('./data/icu_hadm_dict.json', typ='series').to_dict()
icu_id_sepsis = list(icu2hadm.keys())
hadm_id_sepsis = list(icu2hadm.values())

# PREPROCESS vitals data to keep only sepsis cohort and to map icustay_id to hadm_id
vitals_df = pd.read_csv('./data/pivoted_vitals.csv')
vitals_df = vitals_df[vitals_df['icustay_id'].isin(icu_id_sepsis)]
vitals_df = vitals_df.replace({'icustay_id': icu2hadm})
vitals_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
vitals_df['charttime'] = vitals_df['charttime'].str.replace('\S{6}$', ':00:00', regex=True)
vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'], format='%Y-%m-%d %H:%M:%S')
vitals_df.to_csv('./data/processed_pivoted_vitals.csv', index=False)
del vitals_df
vitals_df = None

# PREPROCESS labs data to keep only sepsis cohort
labs_df = pd.read_csv('./data/pivoted_labs.csv')
labs_df = labs_df[labs_df['hadm_id'].isin(hadm_id_sepsis)]
labs_df.to_csv('./data/processed_pivoted_labs.csv', index=False)
del labs_df
labs_df = None

# PREPROCESS gcs data to keep only sepsis cohort and to map icustay_id to hadm_id
gcs_df = pd.read_csv('./data/pivoted_gcs.csv')
gcs_df = gcs_df[gcs_df['icustay_id'].isin(icu_id_sepsis)]
gcs_df = gcs_df.replace({'icustay_id': icu2hadm})
gcs_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
gcs_df.to_csv('./data/processed_pivoted_gcs.csv', index=False)
del gcs_df
gcs_df = None

# PREPROCESS urine data to keep only sepsis cohort and to map icustay_id to hadm_id
uo_df = pd.read_csv('./data/urine_output.csv')
uo_df = uo_df[uo_df['icustay_id'].isin(icu_id_sepsis)]
uo_df = uo_df.replace({'icustay_id': icu2hadm})
uo_df.rename(columns={'icustay_id':'hadm_id'}, inplace=True)
uo_df.to_csv('./data/processed_urine_output.csv', index=False)
del uo_df
uo_df = None