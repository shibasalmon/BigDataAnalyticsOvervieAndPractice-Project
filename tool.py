import numpy    as np
import pandas   as pd

yi = pd.read_excel('AQI總表.xlsx')['AQI大於100之比率(%)'][:2376]
f = 0.0
for a in [5.0, 10.0, 20.0]:
    for i in range(yi.shape[0]):
        print(yi[i])
        if (yi[i]-a) > 0:
            yi[i] = 1
        else:
            yi[i] = 0
        print(yi[i])

    pd.DataFrame(yi).to_csv('MC' +str(int(a)) +'.csv')