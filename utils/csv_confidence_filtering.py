import pandas as pd
import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


csv_path="/opt/ml/level2_objectdetection_cv-level2-cv-13/Ensemble/temp/Cascade_fold5.csv"
confidence_threshold=0.3


df=pd.read_csv(csv_path)
df_save=pd.read_csv(csv_path)

for i in range(len(df)):
    s=df.iloc[i]['PredictionString']
    df_save['PredictionString'][i]=""
    if type(s)!=str:
        continue
    s=s.split()
    annotation=[]
    for j in range(0,len(s),6):
        label,confidence,min_x,min_y,max_x,max_y=s[j],s[j+1],s[j+2],s[j+3],s[j+4],s[j+5]
        if float(confidence)>confidence_threshold:
            annotation+=s[j:j+6]
    df_save['PredictionString'][i]=" ".join(map(str,annotation))
df_save.to_csv('/opt/ml/level2_objectdetection_cv-level2-cv-13/Ensemble/temp/fold5_cascade_0.3.csv')
        