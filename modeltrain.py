#%%

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

def trainModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    # ROC_Curve를 그리기 위해 predict_proba를 구한다
    train_proba = model.predict_proba(X_train)[:,1]
    train_pred = [1 if p>0.5 else 0 for p in train_proba]

    test_proba = model.predict_proba(X_test)[:,1]
    test_pred = [1 if p>0.5 else 0 for p in test_proba]

    # precision, recall, f1-score를 다 확인할 수 있음
    trainReport = classification_report(y_train, train_pred)
    testReport = classification_report(y_test, test_pred)
    
    # ROC_AUC_SCORE : 얼마나 예측을 잘했는지 확인하는 지표
    trainRoc = roc_auc_score(y_train, train_proba)
    testRoc = roc_auc_score(y_test, test_proba)

    print(trainReport)
    print(testReport)
    print(f'trainRoc:{trainRoc}')
    print(f'testRoc:{testRoc}')

    return test_pred

# %%
