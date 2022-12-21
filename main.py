#%%
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from dataprep.eda import create_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pymysql
from sqlalchemy import create_engine

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost
# %%
df = pd.read_csv(r'.\data\customer.csv')
dbpath = 'mysql+pymysql://root:276018@localhost/telecom'
conn = create_engine(dbpath)
df.to_sql(name='telecomdb', con=conn, if_exists='fail', index=False)
# %%
conn = pymysql.connect(host='localhost', user='root', passwd=str(276018), database='telecom', charset='utf8')
df = pd.read_sql('select * from telecomdb', con=conn)
# %%
print(f"df shape :{df.shape}")
print(f"df info: {df.info()}")
df.head()
# %%
# 문자 데이터와 숫자 데이터를 나눔
obj_df = df.select_dtypes('object')
num_df = df.select_dtypes('number')
obj_cols = list(obj_df)
num_cols = list(num_df)

print(f"length of obj_cols: {len(obj_cols)}")
print(f"length of num_cols: {len(num_cols)}")
# %%
# TotalCharges (object -> num)
# SeniorCitizen (num -> object)
df['TotalCharges'].value_counts().index # value unique값 확인
df[df['TotalCharges'] == ' '] = None
df['TotalCharges'].value_counts(dropna=False)
df.dropna(axis=0, inplace=True)
# %%
df['TotalCharges'] = df['TotalCharges'].astype('float')
df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
# %%
obj_df = df.select_dtypes('object')
num_df = df.select_dtypes('number')
obj_cols = list(obj_df)
num_cols = list(num_df)
# %%
print(obj_cols)
print(num_cols)
# %%
# 시각화
# 연속형 데이터는 distplot, 범주형 데이터는 histplot
plt.figure(figsize=(12,10))
