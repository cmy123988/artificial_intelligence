import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target

#样本数和特征数
n_samples = X_full.shape[0]
n_features = X_full.shape[1]
#首先确定我们希望放入的缺失数据的比例，在这里我们假设是50%
rng = np.random.RandomState(0)
#缺失率
missing_rate = 0.5
#应确实总数
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
#通过随机数设置这n_missing_samples个缺失值的横纵索引
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)
#保留原数据
X_missing = X_full.copy()
y_missing = y_full.copy()
#设置缺失值np.nan并转换为DataFrame格式
X_missing[missing_samples, missing_features] = np.nan
X_missing = pd.DataFrame(X_missing)

#设置SimpleImputer利用均值填充
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_missing_mean = imp_mean.fit_transform(X_missing)

#设置SimpleImputer利用0填充
imp_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
X_missing_0 = imp_0.fit_transform(X_missing)

X_missing_reg = X_missing.copy()
#利用np.argsort将数据按每列的缺失值进行排序并获取排序索引
sortindex = np.argsort(X_missing_reg.isnull().sum(axis=0)).values
#依次遍历每个索引
for i in sortindex:
    #构建我们的新特征矩阵和新标签
    df = X_missing_reg
    #需要填补的列
    fillc = df.iloc[:,i]
    #将除i以外的列以及房价y列合并作为训练集
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
    #将训练集缺失值部分取0
    df_0 = imp_0.fit_transform(df)

    #划分训练测试集
    Ytrain = fillc[fillc.notnull()]
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]
    #构造随机森林
    rfc = RandomForestRegressor(n_estimators=25)
    rfc.fit(Xtrain, Ytrain)
    #预测并将值保存到原始矩阵中
    Ypredict = rfc.predict(Xtest)
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict

#进行建模分析
X = [X_full, X_missing_mean, X_missing_0, X_missing_reg]
mse = []
std = []
#一次对三种情况与原始数据进行预测评估
for x in X:
    estimator = RandomForestRegressor(random_state=0, n_estimators=50)
    score = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=5).mean()
    #注意上述score为负MSE，需要×-1
    mse.append(score * -1)

#可视化
x_labels = ['Full data', 'Zero Imputation', 'Mean Imputation', 'Regressor Imputation']
colors = ['r', 'g', 'b', 'orange']
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
for i in np.arange(len(mse)):
	#水平条形图将数据依次遍历画出
    ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')
ax.set_title('Imputation Techniques with Boston Data')
#设置x轴坐标上下限
ax.set_xlim(left=np.min(mse) * 0.9, right=np.max(mse) * 1.1)
#y轴刻度
ax.set_yticks(np.arange(len(mse)))
#x轴标签
ax.set_xlabel('MSE')
#y轴每个刻度的标签
ax.set_yticklabels(x_labels)
plt.show()
