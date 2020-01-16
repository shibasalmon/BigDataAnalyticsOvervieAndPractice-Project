import numpy    as np
import pandas   as pd


def FeatureScaling(x):
    for i in range(1, x.shape[1]):
        x[:,i] = x[:,i] - x[:,i].mean()
        x[:,i] = x[:,i] / x[:,i].max()

def hy(x, theta):
    tmp = 1 + np.exp(-x.dot(theta))
    result = 1 / tmp
    return result

def CostFunc(h, y):
    result = y*np.log10(h) + (1-y)*np.log10(1-h)
    return result

def Error(x, theta, y):
    m = x.shape[0]
    h = hy(x, theta)
    J = CostFunc(h, y).sum() / -m
    return J

def LogisticRegression(x, y):
    m = x.shape[0]
    n = x.shape[1]
    theta = np.ones(x.shape[1])
    alpha = 1
    preError = 0
    while abs(Error(x, theta, y) - preError)*10000000 > 0.0000000001:
        new_theta = np.ones(n)
        for j in range(n):
            tmp = ((hy(x, theta) - y)*x[:, j]).sum() * alpha / m
            new_theta[j] = theta[j] - tmp

        preError = Error(x, theta, y)
        theta = new_theta
        # print(Error(x, theta, y))
        # print(theta)
        
    return theta

df1 = pd.read_csv('天氣總表.csv')
# df2 = pd.read_excel('火力發電總表.xlsx')
# df2 = df2[['年份', '月份', '新北市', '台中市', '高雄市', '花蓮縣', '雲林縣', '桃園市', '苗栗縣', '澎湖縣', '金門縣', '連江縣']]
yi = pd.read_csv('for_MC.csv')['AQI大於100之比率(%)'][:2376]


xi = df1[['Temperature','Precp','RH','StnPres','WS']]
xi.insert(0, 'Constant', 1)
# x['T^2'] = df1['Temperature'] * df1['Temperature']
# x['P^2'] = df1['Precp'] * df1['Precp']
# x['R^2'] = df1['RH'] * df1['RH']
# x['S^2'] = df1['StnPres'] * df1['StnPres']
# x['W^2'] = df1['WS'] * df1['WS']
x = xi.to_numpy()
y = yi.to_numpy()

FeatureScaling(x)
theta = LogisticRegression(x, y)

print(theta)
df = pd.DataFrame([hy(x, theta), y])
df.to_csv('result.csv')
# result = pd.DataFrame(h, y)
# result.to_csv('result.csv')