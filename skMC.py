import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics import confusion_matrix

LR = LogisticRegression()

df1 = pd.read_csv('天氣總表.csv')
yi = pd.read_csv('for_MC.csv')['AQI大於100之比率(%)'][:2376]

xi = df1[['Temperature','Precp','RH','StnPres','WS']]
xi.insert(0, 'Constant', 1)

# 用來增加參數
xi['T^2'] = df1['Temperature'] * df1['Temperature']
xi['P^2'] = df1['Precp'] * df1['Precp']
xi['R^2'] = df1['RH'] * df1['RH']
xi['S^2'] = df1['StnPres'] * df1['StnPres']
xi['W^2'] = df1['WS'] * df1['WS']
xi['TP'] = df1['Temperature'] * df1['Precp']
xi['TR'] = df1['Temperature'] * df1['RH']
xi['TS'] = df1['Temperature'] * df1['StnPres']
xi['TW'] = df1['Temperature'] * df1['WS']
xi['PR'] = df1['Precp'] * df1['RH']
xi['PS'] = df1['Precp'] * df1['StnPres']
xi['PW'] = df1['Precp'] * df1['WS']
xi['RS'] = df1['RH'] * df1['StnPres']
xi['RW'] = df1['RH'] * df1['WS']
xi['SW'] = df1['StnPres'] * df1['WS']
# ====================================

x = xi.to_numpy()
y = yi.to_numpy(dtype=int)

sc = StandardScaler()
sc.fit(x)
x = sc.fit(x).transform(x)

LR.fit(x, y)

# =============
xt = x
yt = LR.predict(xt)
accuracy = LR.score(x, y)
tn, fp, fn, tp = confusion_matrix(y, yt).ravel()
recall = tp / (tp + fn)

print(str(accuracy) + ' ' + str(recall))

df = pd.DataFrame(y)
df.to_csv('result.csv')