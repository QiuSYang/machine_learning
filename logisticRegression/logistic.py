"""
Logistic 回归 概述
Logistic 回归 或者叫逻辑回归 虽然名字有回归，但是它是用来做分类的。
其主要思想是: 根据现有数据对分类边界线(Decision Boundary)建立回归公式，以此进行分类。

基于最优化方法的回归系数确定
Sigmoid 函数的输入记为 z ，由下面公式得到:
    z = w0x0 + w1x1 + w2x2 + ... + wnxn
如果采用向量的写法，上述公式可以写成 z = w^Tx ，它表示将这两个数值向量对应元素相乘然后全部加起来即得到 z 值。
其中的向量 x 是分类器的输入数据，向量 w 也就是我们要找到的最佳参数（系数），从而使得分类器尽可能地精确。
为了寻找该最佳参数，需要用到最优化理论的一些知识。我们这里使用的是——梯度上升法（Gradient Ascent）。

问：
有人会好奇为什么有些书籍上说的是梯度下降法（Gradient Decent）?
答： 
其实这个两个方法在此情况下本质上是相同的。关键在于代价函数（cost function）或者叫目标函数（objective function）。
如果目标函数是损失函数，那就是最小化损失函数来求函数的最小值，就用梯度下降。 
如果目标函数是似然函数（Likelihood function），就是要最大化似然函数来求函数的最大值，那就用梯度上升。
在逻辑回归中， 损失函数和似然函数无非就是互为正负关系。

Logistic 回归 原理
Logistic 回归 工作原理
每个回归系数初始化为 1
重复 R 次:
    计算整个数据集的梯度
    使用 步长 x 梯度 更新回归系数的向量
返回回归系数

Logistic 回归 开发流程
收集数据: 采用任意方法收集数据
准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
分析数据: 采用任意方法对数据进行分析。
训练算法: 大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
测试算法: 一旦训练步骤完成，分类将会很快。
使用算法: 首先，我们需要输入一些数据，并将其转换成对应的结构化数值；
        接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；
        在这之后，我们就可以在输出的类别上做一些其他分析工作。

Logistic 回归 算法特点
优点: 计算代价不高，易于理解和实现。
缺点: 容易欠拟合，分类精度可能不高。
适用数据类型: 数值型和标称型数据。
"""
import os 
import sys 
import time 
import random 
import numpy as np 
import matplotlib.pyplot as plt 

class SELogistic(object):
    def __init__(self):
        pass 
    
    # 解析数据
    def loadDataSet(self, file_name):
        '''
        Desc: 
            加载并解析数据
        Args:
            file_name -- 要解析的文件路径
        Returns:
            dataMat -- 原始数据的特征
            labelMat -- 原始数据的标签，也就是每条样本对应的类别。即目标向量
        '''
        # dataMat为原始数据, labelMat为原始数据的标签
        dataMat = []
        labelMat = []
        fr = open(file_name)
        if not fr:
            raise ValueError("file open failure.")
        
        for lineContent in fr.readlines():
            lineArr = lineContent.strip().split('\t')
            # 为方便计算，我们将x0的值设置为1.0, 也就是在每一行的开头添加一个1.0作为x0
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))

        return dataMat, labelMat
    
    # sigmoid阶跃函数
    def sigmoid(self, inX):
        # return 1.0 / (1 + exp(-inX))

        # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。
        # 因此，实际应用中，tanh 会比 sigmoid 更好。
        return 2 * 1.0/(1 + np.exp(-2*inX)) - 1
    
    # 正常的处理方案
    # 两个参数：第一个参数==> dataMatIn 是一个2维NumPy数组，每列分别代表每个不同的特征，每行则代表每个训练样本。
    # 第二个参数==> classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，
    # 做法是将原向量转置，再将它赋值给labelMat。
    def gradAscent(self, dataMatIn, classLabels):
        # 转化为矩阵[[1,1,2],[1,1,2]....]
        dataMatrix = np.mat(dataMatIn) # 转化为numpy矩阵
        # 转化为矩阵[[0,1,0,1,0,1.....]]，并转制[[0],[1],[0].....]
        # transpose() 行列转置函数
        # 将行向量转化为列向量   =>  矩阵的转置 
        # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
        labelMat = np.mat(classLabels).transpose() 
        #labelMat = np.array(classLabels).reshape(-1, 1)
        #print(labelMat)
        # m->数据量，样本数 n->特征数
        m, n = dataMatrix.shape
        # alpha代表向目标移动的步长
        alpha = 0.01
        # 迭代次数
        maxCycles = 1000
        # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
        # weights 代表回归系数, 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
        weights = np.ones((n,1))
        for k in range(maxCycles):
            # m*3 的矩阵 * 3*1 的矩阵 ＝ m*1的矩阵
            # 那么乘上矩阵的意义，就代表：通过公式得到的理论值
            # 参考地址： 矩阵乘法的本质是什么？ https://www.zhihu.com/question/21351965/answer/31050145
            # print 'dataMatrix====', dataMatrix 
            # print 'weights====', weights
            # n*3   *  3*1  = n*1
            h = self.sigmoid(dataMatrix * weights) # 矩阵乘法
            # 实际值与预测值之间的差
            error = labelMat - h # 列向量相减
            # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，
            # 最后得出 x1,x2,...xn的系数的偏移量
            weights = weights + alpha * dataMatrix.transpose() * error # 矩阵乘法，最后得到回归系数

        return np.array(weights)
    
    """
    注意
    梯度上升算法在每次更新回归系数时都需要遍历整个数据集，该方法在处理 100 个左右的数据集时尚可，
    但如果有数十亿样本和成千上万的特征，那么该方法的计算复杂度就太高了。
    一种改进方法是一次仅用一个样本点来更新回归系数，该方法称为 随机梯度上升算法。
    由于可以在新样本到来时对分类器进行增量式更新，因而随机梯度上升算法是一个在线学习(online learning)算法。
    与 “在线学习” 相对应，一次处理所有数据被称作是 “批处理” （batch） 。

    随机梯度上升算法可以写成如下的伪代码:

    所有回归系数初始化为 1
    对数据集中每个样本
        计算该样本的梯度
        使用 alpha x gradient 更新回归系数值
    返回回归系数值
    """
    # 随机梯度上升
    # 梯度上升优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
    # 随机梯度上升一次只用一个样本点来更新回归系数
    def stocGradAscent_(self, dataMatrix, classLabels):
        '''
            dataMatrix: np.array()矩阵
        '''
        m, n = dataMatrix.shape
        alpha = 0.01 
        weights = np.ones(n)
        for i in range(m):
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,
            # 此处求出的 h 是一个具体的数值，而不是一个矩阵
            h = self.sigmoid(sum(dataMatrix[i] * weights))
            # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
            error = classLabels[i] - h
            print("weights is {}, the i_th data is {}, error is {}".format(weights, dataMatrix[i], error))
            weights = weights + alpha * error * dataMatrix[i]
        
        return weights

    # 随机梯度上升算法(随机化)
    def stocGradAscent__(self, dataMatrix, classLabels, numIter=150):
        m, n = dataMatrix.shape
        print(m, n)
        print(type(dataMatrix[0, 0]))
        weights = np.ones(n) 
        # 随机梯度下降， 迭代循环numIter, 观察是否收敛
        for j in range(numIter):
            dataIndex = list(range(m)) # [0, 1, 2, ... , m-1]
            for i in range(m):
                # i和j的不断增大，导致alpha的值不断减少，但是不为0
                # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
                alpha = 4/(1.0 + j + i) + 0.0001
                # 随机产生一个 0～len()之间的一个值
                # random.uniform(x, y) 方法将随机生成下一个实数，
                # 它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
                randIndex = int(random.uniform(0, len(dataIndex)))
                # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
                h = self.sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
                error = classLabels[dataIndex[randIndex]] - h 
                #print("weights is {}, the i_th data is {}, error is {}".format(weights, dataMatrix[dataIndex[randIndex]], error))
                weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
                # 删除已经使用过的数据索引
                del(dataIndex[randIndex])
        return weights


    # 分类函数，根据回归系数和特征向量来计算 Sigmoid的值
    def classifyVector(self, inX, weights):
        '''
        Desc: 
            最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
        Args:
            inX -- 特征向量，features
            weights -- 根据梯度下降/随机梯度下降 计算得到的回归系数
        Returns:
            如果 prob 计算大于 0.5 函数返回 1
            否则返回 0
        '''
        prob = self.sigmoid(sum(inX * weights))
        if prob > 0.5:
            return 1.0 
        else:
            return 0.0 
    
    # 打开测试集和训练集,并对数据进行格式化处理
    def colicTest(self):
        '''
        Desc:
            打开测试集和训练集，并对数据进行格式化处理
        Args:
            None
        Returns:
            errorRate -- 分类错误率
        '''
        frTrain = open('./datasets/horseColicTraining.txt')
        frTest = open('./datasets/horseColicTest.txt')
        if not frTrain or not frTest:
            raise ValueError("open file is failure.")
        trainingSet = []
        trainingLabels = []
        # 解析训练数据集中的数据特征和Labels
        # trainingSet 中存储训练数据集的特征，
        # trainingLabels 存储训练数据集的样本对应的分类标签
        for line in frTrain.readlines():
            currentLine = line.strip().split('\t')
            trainingSet.append(currentLine[:-1])
            trainingLabels.append(currentLine[-1])
        
        # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
        '''
        trainWeights = self.stocGradAscent__(np.array(trainingSet, dtype=np.float), 
                            np.array(trainingLabels, dtype=np.float), 500) 
        '''

        trainWeights = self.gradAscent(np.array(trainingSet, dtype=np.float), 
                            np.array(trainingLabels, dtype=np.float))
        trainWeights = trainWeights.reshape(1, -1)[0]
        print(trainWeights)
        errorCount = 0 
        numTestVec = 0.0 
        # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
        for line in frTest.readlines():
            numTestVec += 1.0 
            currentLine = line.strip().split('\t')
            if int(self.classifyVector(np.array(currentLine[:-1], dtype=np.float), trainWeights)) != int(currentLine[-1]):
                errorCount += 1
        
        errorRate = float(errorCount)/numTestVec
        print("the error rate of this test is: {}".format(errorRate))
        
        return errorRate

    # 画出数据集和 Logistic 回归最佳拟合直线的函数
    def plotBestFit(self, dataArr, labelMat, weights):
        '''
            Desc:
                将我们得到的数据可视化展示出来
            Args:
                dataArr:样本数据的特征
                labelMat:样本数据的类别标签，即目标变量
                weights:回归系数
            Returns:
                None
        '''
        n = dataArr.shape[0] # 获取数据量
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []

        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i, 1])
                ycord1.append(dataArr[i, 2])
            else:
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label='class1')
        ax.scatter(xcord2, ycord2, s=30, c='green', label='class0')
        x = np.arange(-3.0, 3.0, 0.1)
        """
        y的由来，卧槽，是不是没看懂？
        首先理论上是这个样子的。
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        w0*x0+w1*x1+w2*x2=f(x)
        x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
        所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
        """
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    '''
    dataPath = './datasets/TestSet.txt'
    pt = SELogistic()
    datas, labels = pt.loadDataSet(dataPath)
    print(datas[1][:])
    plt.scatter(np.array(datas)[:, 1], np.array(datas)[:, 2], c=np.array(labels)*5)
    plt.show()

    #weights = pt.gradAscent(np.array(datas), labels)
    #weights = pt.stocGradAscent_(np.array(datas), labels)
    weights = pt.stocGradAscent__(np.array(datas), labels, numIter=10000)
    print(weights)
    # 数据可视化
    pt.plotBestFit(np.array(datas), labels, weights)
    '''

    pt = SELogistic()
    # 测试10次求平均结果
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += pt.colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

