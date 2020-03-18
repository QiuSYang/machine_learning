"""
pandas base operation for the data analyze
"""
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def pandasStatic():
    # 生产数据, 第一行为1-7， 第二行为2-8
    D = pd.DataFrame([range(1, 8), range(2, 9)])
    print("origin data is {}".format(D))
    # 计算相关系数矩阵
    print(D.corr(method='pearson'))
    # 提取第一行
    S1 = D.loc[0]
    print(S1)
    # 提取第二行
    S2 = D.loc[1]
    print(S2)
    # 计算 S1 和 S2的相关系数
    print(S1.corr(S2, method='pearson'))

    # 计算数据样本的偏度和峰度
    # 生产一个6x5的随机矩阵
    D = pd.DataFrame(np.random.randn(6, 5))
    print(D)
    # 计算协方差矩阵
    print(D.cov)
    # 计算偏度
    print(D.skew())
    # 计算峰度
    print(D.kurt())

def pltBase():
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    plt.figure(figsize=(7, 5)) # 创建图像区域，指定比例

    x = np.linspace(0, 2*np.pi, 50)
    y = np.sin(x)
    # 控制图形格式为蓝色带星虚线，显示正弦曲线
    plt.plot(x, y, 'bp--')
    plt.show()
    plt.close()

    # 绘制饼状图
    labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
    # 每块比例
    sizes = [15, 30, 45, 10]
    # 每块的颜色
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    # 突出显示，这里仅仅突出显示第二块
    explode = (0, 0.1, 0, 0)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal') #显示为圆(避免比例压缩为椭圆)
    plt.show()
    plt.close()

    # 绘制箱型图
    x = np.random.randn(1000) # 1000个服从正态分布的随机数
    # 构造两列的DataFrame
    D = pd.DataFrame([x, x+1]).T 
    # 调用Series内置的作图方法画图，用kind参数指定箱型图box
    D.plot(kind='box')
    plt.show()
    plt.close() 

    # 绘制误差线
    error = np.random.randn(10) # 定义误差列
    y = pd.Series(np.sin(np.arange(10))) # 均值数据列
    y.plot(yerr=error) # 绘制误差图 
    plt.show() 


if __name__ == '__main__':
    #pandasStatic()
    pltBase()