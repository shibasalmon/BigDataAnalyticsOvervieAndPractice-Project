import numpy    as np
import pandas   as pd

def Error(x, theta, y):
    m = x.shape[0]
    J = ((x.dot(theta) - y)*(x.dot(theta) - y)).sum() / 2 / m
    return J

def GradientDesent(x, theta, y):
    m = x.shape[0]
    n = x.shape[1]
    alpha = 0.1
    while Error(x, theta, y) > 30:
        new_theta = np.ones(n)
        for j in range(n):
            tmp = ((x.dot(theta) - y)*x[:, j]).sum() * alpha / m
            new_theta[j] = theta[j] - tmp

        theta = new_theta
        print(Error(x, theta, y))
        print(theta)

def FeatureScaling(x):
    for i in range(1, x.shape[1]-1):
        x[:,i] = (x[:,i] - x[:,i].mean())/x[:,i].max()

df1 = pd.read_csv('天氣總表.csv')
# df2 = pd.read_excel('火力發電總表.xlsx')
# df2 = df2[['年份', '月份', '新北市', '台中市', '高雄市', '花蓮縣', '雲林縣', '桃園市', '苗栗縣', '澎湖縣', '金門縣', '連江縣']]
yi = pd.read_excel('AQI總表.xlsx')['AQI大於100之比率(%)'][:2376]


x = df1[['Temperature','Precp','RH','StnPres','WS']]
x.insert(0, 'Constant', 1)
# x['T^2'] = df1['Temperature'] * df1['Temperature']
# x['P^2'] = df1['Precp'] * df1['Precp']
# x['R^2'] = df1['RH'] * df1['RH']
# x['S^2'] = df1['StnPres'] * df1['StnPres']
# x['W^2'] = df1['WS'] * df1['WS']
x = x.to_numpy()
y = yi.to_numpy()
theta = np.ones(x.shape[1])

FeatureScaling(x)
GradientDesent(x, theta, y)

h = x.dot(theta)
result = pd.DataFrame(h, y)
result.to_csv('result.csv')