import numpy    as np
import pandas   as pd

df1 = pd.read_csv('天氣總表.csv')
# df2 = pd.read_excel('火力發電總表.xlsx')
# df2 = df2[['年份', '月份', '新北市', '台中市', '高雄市', '花蓮縣', '雲林縣', '桃園市', '苗栗縣', '澎湖縣', '金門縣', '連江縣']]
yi = pd.read_excel('AQI總表.xlsx')['AQI大於100之比率(%)'][:2376]


x = df1[['Temperature','Precp','RH','StnPres','WS']]
x['Constant'] = pd.Series()
x['Constant'] = 1
# x['T^2'] = df1['Temperature'] * df1['Temperature']
# x['P^2'] = df1['Precp'] * df1['Precp']
# x['R^2'] = df1['RH'] * df1['RH']
# x['S^2'] = df1['StnPres'] * df1['StnPres']
# x['W^2'] = df1['WS'] * df1['WS']
x = x.to_numpy()
y = yi.to_numpy()

theta = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
print(theta)

h = x.dot(theta)
result = pd.DataFrame(h, y)
result.to_csv('result.csv')