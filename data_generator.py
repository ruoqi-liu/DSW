import pandas as pd
import json

sepsis3_df = pd.read_csv('./data/sepsis3-df.csv')

icustay_ids = sepsis3_df['icustay_id']
icustay_ids.to_csv('./data/icustay_ids.txt', index=False)

icu2hadm_df = sepsis3_df[['icustay_id', 'hadm_id']]
icu2hadm_dict = icu2hadm_df.set_index('icustay_id').to_dict()['hadm_id']
icu2hadm_json = json.dumps(icu2hadm_dict)
dict_file = open('./data/icu_hadm_dict.json', "w")
dict_file.write(icu2hadm_json)
dict_file.close()
