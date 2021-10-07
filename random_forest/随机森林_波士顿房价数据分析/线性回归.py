import pandas as pd
from sklearn.datasets import load_boston  # 波士顿房价数据
from sklearn.model_selection import train_test_split  # 拆分数据集
from sklearn.preprocessing import StandardScaler  # 标准差标准化
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.linear_model import SGDRegressor  # 线性回归模型
from sklearn.linear_model import Ridge  # 岭回归
import matplotlib.pyplot as plt
import numpy as np


def show_res(y_true, y_predict):
    """
    结果显示
    :param y_true: 真实房价
    :param y_predict: 预测房价
    :return: None
    """
    # 1、创建画布

    plt.figure()
    # 默认不支持中文
    # 修改RC参数，来让其支持中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # 2、绘图
    x = np.arange(1, y_predict.shape[0] + 1)
    # 真实值的走势
    plt.plot(x, y_true, marker="*", color="g", linestyle=":", markersize=4)
    # 预测值的走势
    plt.plot(x, y_predict, marker="o", color="pink", markersize=4)

    # 增加图例
    plt.legend(["真实房价", "预测房价"])
    # 增加标题
    plt.title("波士顿房价真实与预测房价走势图")

    # 保存图片---如果在show之后保存图片，那么图片是完全空白的
    plt.savefig("./波士顿房价真实与预测房价走势图.png")
    # 3、展示
    plt.show()


# 1、加载数据
boston_data = load_boston()
# print("boston_data:\n", boston_data)

# 2、获取特征值、获取目标值、获取特征名称
feature = boston_data["data"]
print("feature:\n", feature)
print("feature 的形状:\n", feature.shape)

#
target = boston_data["target"]
print("target:\n", target)
print("target 的形状:\n", target.shape)

#
feature_names = boston_data["feature_names"]
print("feature_names:\n", feature_names)
print("feature_names 的形状:\n", feature_names.shape)
print("*" * 100)

# 3、可以将boston_data 保存到本地
# df.to_xxx pandas 中数据保存形式
# 将特征值转化为df
# df_feature = pd.DataFrame(data=feature, columns=feature_names)
# print("df_feature:\n", df_feature)
# # 将目标值转化为df
# df_target = pd.DataFrame(data=target, columns=["MEDV"])
# print("df_target:\n", df_target)
# print("*" * 100)
#
# # 拼接特征值与目标值
# res_data = pd.concat((df_feature, df_target), axis=1)
# print("res_data:\n", res_data)
#
# # 保存数据
# res_data.to_excel("./boston_data.xlsx", index=False)

# 3、拆分数据集
# 之前 手动拆分 data 为完整数据集：   data[:400,:] 训练集  data[400:,:]测试集
# 拆分数据集
# 参数1  特征值
# 参数2 目标值
# 参数3 test_size ---测试集占比
# 特征值（训练集的特征值、测试集的特征值）   目标值（训练集的目标值、测试集的目标值）
# 随机拆分---默认的
# random_state ---给定值，把数据集拆分固定
# 如果要进行超参数优化，数据集必须固定
# 训练集：测试集  = 7:3
# 训练集、验证集、测试集 = 8:1:1
# 验证集----验证超参数
train_x, test_x, train_y, test_y = train_test_split(feature, target, test_size=0.3, random_state=1)
print("训练集的特征值:\n", train_x)
print("训练集的特征值:\n", train_x.shape)
print("训练集的目标值:\n", train_y)
print("训练集的目标值:\n", train_y.shape)
print("测试集的特征值:\n", test_x)
print("测试集的特征值:\n", test_x.shape)
print("测试集的目标值:\n", test_y)
print("测试集的目标值:\n", test_y.shape)
print("*" * 100)

# 检测缺失值、处理缺失值 ---这个数据集无缺失值处理
# 处理异常值 ---这个数据集无异常值处理
# 标准化处理----法1
# # （1）实例化对象
stand = StandardScaler()
# # (2)标准化数据
# # 需要标准化哪些数据？？？？--量级相差较大
# # 特征值需要标准化，目标值不需要标准化
# # 线性回归求解---w,b
# # 特征值标准化，目标值不标准化，---w,b,如果得到新的标准化之后的特征值，代入模型，得到的预测值是真实的房价
# # fit_transform ---(x - x.mean() / x.std())
# (1)计算自身的指标（2）进行转化数据
train_x = stand.fit_transform(train_x)
test_x = stand.fit_transform(test_x)

print("标准化之后的数据：\n", train_x)
print("标准化之后的数据：\n", test_x)

# 标准化处理---法2
# (1)实例化对象
# stand = StandardScaler()
# # （2）标准化数据
# # 计算指标
# stand.fit(train_x)
# # 转化
# train_x = stand.transform(train_x)
# # 利用训练集的特征值的指标来转化测试集的特征
# test_x = stand.transform(test_x)


# # 线性回归算法进行房价预测
# # LinearRegression 基于正规方程的求解方式的线性回归
# # 应用于数据较小，特征较少，模型构建不复杂的情况下
# (1)构建算法实例
lr = LinearRegression()
# （2）训练数据
lr.fit(train_x, train_y)
# (3)预测数据
y_predict = lr.predict(test_x)

# 获取准确率
score = lr.score(test_x, test_y)

# 获取权重与偏置
weight = lr.coef_
bias = lr.intercept_

print("准确率为：\n", score)
print("权重为：\n", weight)
print("偏置为：\n", bias)
print("预测值：\n", y_predict)

# (1)构建算法实例
# 用于数据量较大、特征较多、模型较大的情况下
# 随机梯度下降优化算法进行求解w,b
# 梯度方向---随机的
# 学习率---
# penalty= "l2" 正则化--L2正则化，alpha--正则化力度
# learning_rate = "invscaling" --默认学习率
# 想要更改学习率
# （1）将learning_rate ="constant",（2）再去更改eta0的值
# 更改的学习率：不能过大，可能会造成梯度爆炸现象--会出现NaN的结果，
# 也不能过小，会造成梯度消失，只训练而损失与准确率不变的情况
# sgd = SGDRegressor()
# # （2）训练数据
# sgd.fit(train_x, train_y)
# # (3)预测数据
# y_predict = sgd.predict(test_x)
#
# # 获取准确率
# score = sgd.score(test_x, test_y)
#
# # 获取权重与偏置
# weight = sgd.coef_
# bias = sgd.intercept_
#
# print("准确率为：\n", score)
# print("权重为：\n", weight)
# print("偏置为：\n", bias)
# print("预测值：\n", y_predict)


# (1)构建算法实例
# 线性回归 + L2正则化  ---岭回归
# 数据量较小，特征较少、模型不复杂的情况，也可以使用岭回归
# rd = Ridge()
# # （2）训练数据
# rd.fit(train_x, train_y)
# # (3)预测数据
# y_predict = rd.predict(test_x)
#
# # 获取准确率
# score = rd.score(test_x, test_y)
#
# # 获取权重与偏置
# weight = rd.coef_
# bias = rd.intercept_
#
# print("准确率为：\n", score)
# print("权重为：\n", weight)
# print("偏置为：\n", bias)
# print("预测值：\n", y_predict)

# 增加可视化---看真实值的走势 与预测值之间走势
show_res(test_y, y_predict)