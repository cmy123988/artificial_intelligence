import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import load_boston
import seaborn as sns
from sklearn.datasets import load_boston
boston = load_boston()

feature_names = boston['feature_names']
feature_names
data = boston['data']
#房价单位为1000美元
prices = boston['target']

#数据提取
data_df = pd.DataFrame(data,columns=feature_names)
prices_df = pd.DataFrame(prices,columns=['MEDV'])
boston_df = pd.DataFrame(data_df)
boston_df['MEDV'] = prices
boston_df.describe()

# boston_df.info()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
boston_scaler = pd.DataFrame(boston_df)
for feature in boston_df.columns:
    boston_scaler[feature] = scaler.fit_transform(boston_df[[feature]])
boston_scaler.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = data_df.drop(['MEDV'],axis=1)
y = data_df['MEDV']
# X = boston_scaler.drop(['MEDV'],axis=1)
# y = boston_scaler['MEDV']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators':[5,10,20,50,100,200],#决策树的个数
    'max_depth':[3,5,7],#最大树深
    'max_features':[0.6,0.7,0.8,1]
}

rf = RandomForestRegressor()

grid = GridSearchCV(rf,param_grid=param_grid,cv=3)

grid.fit(X_train,y_train)

grid.best_params_
rf_reg = grid.best_estimator_
grid.best_score_
rf_reg.feature_importances_
feature_importances = rf_reg.feature_importances_
indices = np.argsort(feature_importances)[::-1]

result = {'label':y_test,'prediction':rf_reg.predict(X_test)}
result = pd.DataFrame(result)

result.head()

result['label'].plot(style='k.',c='orange',markersize=15)
result['prediction'].plot(style='r.',c='green',markersize=15)
plt.legend(fontsize=15,markerscale=1)
plt.tick_params(labelsize=25)
plt.grid()
plt.savefig("随机森林拟合结果.png")
MSE = metrics.mean_squared_error(y,rf_reg.predict(X))
print(MSE)

RMSE = np.sqrt(MSE)
print(RMSE)

print(len(data_df))