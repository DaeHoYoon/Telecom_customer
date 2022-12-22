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
from plot import makedist, makehist

import pymysql
from sqlalchemy import create_engine
from dbmodule import uloaddb, dloaddb

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost
from modeltrain import trainModel
# %%
df = pd.read_csv(r'.\data\customer.csv')
host = 'localhost'
id = 'root'
pw = 276018
dbname = 'telecom'
tbname = 'telecomdb'
#%%
# uloaddb(df, id, pw, host, dbname, tbname)
# %%
df = dloaddb(host, id, pw, dbname, tbname)
# %%
print(f"df shape :{df.shape}")
print(f"df info: {df.info()}")
df.head()
# %%
## 컬럼 정의서
# seniorcitizen : 노인여부
# partner : 결혼여부
# dependents : 부양가족여부
# tenure : 회원 개월수
# phoneservice: 전화서비스여부
# multiplelines : 다회선 여부
# internetservice : 인터넷 서비스 공급자
# onlinesecurity : 온라인 보안 여부
# onlinebackup : 온라인 백업 여부
# deviceprotection : 기기보험 여부
# techsupport : 기술지원 여부
# streamingTV : 스트리밍 TV 여부
# streamingmovies : 스트리밍 영화 여부
# contract : 계약기간
# paperlessbilling : 종이없는 청구 여부
# paymentmethod : 결제수단
# monthlyCharges : 월청구금액
# totalcharges : 청구된 총 금액
# churn : 이탈여부
#%%
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
### 시각화
# 연속형 데이터는 distplot(선까지 같이 그려짐), 범주형 데이터는 histplot
# distplot, histplot 둘 다 분포 확인용
# 연속형 데이터 시각화
figsize=(12,10)
cols = num_cols
nrows = 3
ncols = 1

makedist(df, figsize, cols, nrows, ncols)
# %%
# 범주형 데이터 시각화
figsize=(16,30)
cols = obj_cols
nrows = 9
ncols = 2

makehist(df, figsize, cols, nrows, ncols)
# %%
# 연속형 데이터와 y값인 'churn'의 상관관계를 확인
figsize=(12,10)
cols = num_cols
nrows = 3
ncols = 1
y = 'Churn'

makehist(df, figsize, cols, nrows, ncols, y)
# 확인해 본 결과 tenure(회원 개월 수)가 높으면
# 이탈율이 적은 것 정도를 확인 할 수 있었다.
# %%
# id를 제외한 범주형 컬럼을 레이블링
obj_df = obj_df.drop(['customerID'], axis=1)

le = LabelEncoder()
obj_df = obj_df.apply(le.fit_transform) # 여러개의 컬럼을 레이블링하려면 apply 사용
obj_df.head()
# %%
# 레이블링 된 범주형 컬럼과 연속형 컬럼을 concat
df = pd.concat([obj_df, num_df], axis=1)
print(f'df shape:{df.shape}')
# %%
# 상관관계 그래프를 그려본다
cor = df.corr()
mask = np.zeros_like(cor, dtype=np.bool_)
mask[np.triu_indices_from(mask,1)] = True

fig = plt.figure(figsize=(10,10))
plt.title('Churn correlation')
sns.heatmap(cor, cmap='coolwarm', mask=mask, vmin=-1, vmax=1, annot=True, fmt='.2f', cbar_kws={'shrink':.5})
plt.show()
# %%
# 데이터 분리
X_data = df.drop(['Churn'], axis=1)
y_target = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.2, random_state=0)

print(f'X_train shape:{X_train.shape}')
print(f'y_train shape:{y_train.shape}')
print(f'X_test shape:{X_test.shape}')
print(f'y_test shape:{y_test.shape}')
# %%
# 모델 정의
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
lg_reg = LogisticRegression()
xgb = xgboost.XGBClassifier()
vote = VotingClassifier([('decision', dt_clf),\
    ('random',rf_clf), ('logistic', lg_reg),('xgboost',xgb)], voting='soft')

models = [dt_clf, rf_clf, lg_reg, xgb, vote]
# %%
# 모델 학습
for model in models:
    trainModel(model, X_train, y_train, X_test, y_test)
# %%
# ROC curve 그려보기
for model in models:
    test_pred = trainModel(model, X_train, y_train, X_test, y_test)

    fig = plt.figure(figsize=(10,10))
    tpr, fpr, thr = roc_curve(y_test, test_pred)
    plt.plot(tpr,fpr)
    plt.title(f'ROC CURVE {model.__class__.__name__}', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
# %%
# learning curve 그려보기
for model in models:
    trainsizes, trainscore, testscore = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1,1.0,5), cv=3)

    fig = plt.figure(figsize=(10,10))
    trainMean = np.mean(trainscore, axis=1)
    testMean = np.mean(testscore, axis=1)

    plt.plot(trainsizes, trainMean, '-o', label='train')
    plt.plot(trainsizes, testMean, '-o', label="cross val")
    plt.title(f'{model.__class__.__name__} Learning Curve', size=20)
    plt.xlabel("Train Sizes", fontsize=15)
    plt.ylabel("Score", fontsize=15)
    plt.legend()
# %%
