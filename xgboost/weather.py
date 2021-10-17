##  基础函数库
import numpy as np 
import pandas as pd

## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns

## 我们利用Pandas自带的read_csv函数读取并转化为DataFrame格式
data = pd.read_csv('train.csv')

# ## 利用.info()查看数据的整体信息
# data.info()

# ## 进行简单的数据查看，我们可以利用 .head() 头部.tail()尾部
# data.head()

data = data.fillna(-1)
data.tail()

## 利用value_counts函数查看训练集标签的数量
pd.Series(data['RainTomorrow']).value_counts()

## 对于特征进行一些统计描述
data.describe()

numerical_features = [x for x in data.columns if data[x].dtype == np.float]

category_features = [x for x in data.columns if data[x].dtype != np.float and x != 'RainTomorrow']

# #散点图
# ## 选取三个特征与标签组合的散点可视化
# sns.pairplot(data=data[['Rainfall',
# 'Evaporation',
# 'Sunshine'] + ['RainTomorrow']], diag_kind='hist', hue= 'RainTomorrow')
# plt.show()
# plt.savefig('C:\\Users\\cmy\\Desktop\\xgboost\\img\\scatter_diagram')

# #箱型图
# for col in data[numerical_features].columns:
#     if col != 'RainTomorrow':
#         sns.boxplot(x='RainTomorrow', y=col, saturation=0.5, palette='pastel', data=data)
#         plt.title(col)
#         plt.show()

# #条形图
# tlog = {}
# for i in category_features:
#     tlog[i] = data[data['RainTomorrow'] == 'Yes'][i].value_counts()
# flog = {}
# for i in category_features:
#     flog[i] = data[data['RainTomorrow'] == 'No'][i].value_counts()
# plt.figure(figsize=(10,10))
# plt.subplot(1,2,1)
# plt.title('RainTomorrow')
# sns.barplot(x = pd.DataFrame(tlog['Location']).sort_index()['Location'], y = pd.DataFrame(tlog['Location']).sort_index().index, color = "red")
# plt.subplot(1,2,2)
# plt.title('Not RainTomorrow')
# sns.barplot(x = pd.DataFrame(flog['Location']).sort_index()['Location'], y = pd.DataFrame(flog['Location']).sort_index().index, color = "blue")
# plt.show()

# plt.figure(figsize=(10,2))
# plt.subplot(1,2,1)
# plt.title('RainTomorrow')
# sns.barplot(x = pd.DataFrame(tlog['RainToday'][:2]).sort_index()['RainToday'], y = pd.DataFrame(tlog['RainToday'][:2]).sort_index().index, color = "red")
# plt.subplot(1,2,2)
# plt.title('Not RainTomorrow')
# sns.barplot(x = pd.DataFrame(flog['RainToday'][:2]).sort_index()['RainToday'], y = pd.DataFrame(flog['RainToday'][:2]).sort_index().index, color = "blue")
# plt.show()


#对离散变量进行编码
## 把所有的相同类别的特征编码为同一个值
def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(),
         range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction
for i in category_features:
    data[i] = data[i].apply(get_mapfunction(data[i]))

## 编码后的字符串特征变成了数字

print(data['Location'].unique())


#利用XGBoost进行训练和预测
## 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split

## 选择其类别为0和1的样本 （不包括类别为2的样本）
data_target_part = data['RainTomorrow']
data_features_part = data[[x for x in data.columns if x != 'RainTomorrow']]

## 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 2020)

## 导入XGBoost模型
from xgboost.sklearn import XGBClassifier
## 定义 XGBoost模型 
clf = XGBClassifier()
# 在训练集上训练XGBoost模型
clf.fit(x_train, y_train)

## 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
from sklearn import metrics

## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)

# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


## 从sklearn库中导入网格调参函数
from sklearn.model_selection import GridSearchCV

## 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
subsample = [0.8, 0.9]
colsample_bytree = [0.6, 0.8]
max_depth = [3,5,8]

parameters = { 'learning_rate': learning_rate,
              'subsample': subsample,
              'colsample_bytree':colsample_bytree,
              'max_depth': max_depth}
model = XGBClassifier(n_estimators = 50)

## 进行网格搜索
clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=1,n_jobs=-1)
clf = clf.fit(x_train, y_train)
