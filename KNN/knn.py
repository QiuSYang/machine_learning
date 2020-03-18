"""
KNN 概述：
k-近邻（kNN, k-NearestNeighbor）算法是一种基本分类与回归方法，我们这里只讨论分类问题中的 k-近邻算法。

一句话总结：近朱者赤近墨者黑！

k 近邻算法的输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类。
k 近邻算法假设给定一个训练数据集，其中的实例类别已定。
分类时，对新的实例，根据其 k 个最近邻的训练实例的类别，通过多数表决等方式进行预测。
因此，k近邻算法不具有显式的学习过程。

k 近邻算法实际上利用训练数据集对特征向量空间进行划分，并作为其分类的“模型”。 
k值的选择、距离度量以及分类决策规则是k近邻算法的三个基本要素。

KNN 工作原理：
假设有一个带有标签的样本数据集（训练样本集），其中包含每条数据与所属分类的对应关系。
输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较。
计算新数据与样本数据集中每条数据的距离。
对求得的所有距离进行排序（从小到大，越小表示越相似）。
取前 k （k 一般小于等于 20 ）个样本数据对应的分类标签。
求 k 个数据中出现次数最多的分类标签作为新数据的分类。

简单来说： 通过距离度量来计算查询点（query point）与每个训练数据点的距离，
然后选出与查询点（query point）相近的K个最邻点（K nearest neighbors），
使用分类决策来选出对应的标签来作为该查询点的标签

KNN 通俗理解：
给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，
这 k 个实例的多数属于某个类，就把该输入实例分为这个类。

KNN 开发流程：
收集数据：任何方法
准备数据：距离计算所需要的数值，最好是结构化的数据格式
分析数据：任何方法
训练算法：此步骤不适用于 k-近邻算法
测试算法：计算错误率
使用算法：输入样本数据和结构化的输出结果，然后运行 k-近邻算法判断输入数据分类属于哪个分类，最后对计算出的分类执行后续处理

KNN 算法特点：
优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
适用数据范围：数值型和标称型


"""

import os 
import sys 
import time 
import operator 
import numpy as np 
import matplotlib.pyplot as plt 

class SEKNN(object):
    def __int__(self):
        pass 
    
    def getDatasFromFile(self, fileStr):
        """
        Desc:
            导入训练数据
        parameters:
            fileStr: 数据文件路径
        return: 
            数据矩阵 returnMat 和对应的类别 classLabelVector
        """
        # 打开文件
        fr = open(fileStr)
        if not fr:
            raise ValueError("file open error")
        # 将文件的内容存入列表
        fileContents = fr.readlines()
        # 获取文件中的数据的行数，即特征数
        numberOfLines = len(fileContents)
        # 生产对应的特征空矩阵
        FeatureMat = np.zeros(shape=(numberOfLines, 3))
        classLabelVector = []
        # get datas 
        for index, line in enumerate(fileContents):
            # 移除字符串的头尾
            #print(line)
            line = line.strip()
            # 以‘\t’切割字符串
            lineContents = line.split('\t')
            # 每列的特征数据
            FeatureMat[index, :] = lineContents[:3]
            # 每行的最后一列就是标签数据
            classLabelVector.append(int(lineContents[-1]))
        
        return FeatureMat, classLabelVector
    
    def autoNormal(self, dataSets):
        """
        Desc:
            归一化特征值，消除特征之间量级不同导致的影响
        parameter:
            dataSet: 数据集
        return:
            归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

        归一化公式：
            Y = (X-Xmin)/(Xmax-Xmin)
            其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
        """
        # 计算每个特征的最大值、最小值, 即每列的最大、最小值
        minVals = dataSets.min(axis=0)
        maxVals = dataSets.max(axis=0)
        # 极差
        ranges = maxVals - minVals
        # 归一化操作
        normalFeature = (dataSets - minVals)/ranges
        
        return normalFeature, ranges, minVals
    
    """
    构建KNN算法，算法伪代码
    对于每一个在数据集中的数据点：
    1. 计算目标的数据点（需要分类的数据点）与该数据点的距离
    2. 将距离排序：从小到大
    3. 选取前K个最短距离
    4. 选取这K个中最多的分类类别
    5. 返回该类别来作为目标数据点的预测值
    """
    def classify(self, inX, dataSet, labels, k):
        '''
        inX: 一条测试数据
        dataSet: 训练数据集，即对比数据集
        labels: 训练数据集标签
        k: 选取的K值
        欧氏距离公式：两点之间的距离公式
        d = sqrt(dx*dx + dy*dy)
        '''
        # 获取训练集的大小, 即训练数据集的行数
        trainSize = dataSet.shape[0]
        # 距离度量，欧氏距离
        diffMat = dataSet - inX
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5

        # 获取距离从小到大排序的索引
        sortedDistancesIndexs = distances.argsort()
        # 选取前K个最短距离， 选取这K个中最多的分类类别
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistancesIndexs[i]]
            # dict.get(key, 0)没有关键字key, 就返回0
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 
        # classCount.iteritems()将classCount字典分解为元组列表，
        # operator.itemgetter(1)按照第二个元素的次序对元组进行排序，
        # reverse=True是逆序，即按照从大到小的顺序排列
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # 获取投票最多的类别
        return sortedClassCount[0][0]
    

    # 利用KNN识别手写数字体
    def image2vector(self, filename):
        # 图片数据转为一维向量
        datasVect = np.zeros(shape=(1, 32*32))
        fr = open(filename)
        if not fr:
            raise ValueError("open file is error.")
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                datasVect[0, 32*i+j] = int(lineStr[j])
        
        return datasVect
    
    def handWritingClassifier(self, trainDir, testDir):
        # 1.导入训练数据
        labelsVect = []
        trainFileLists = os.listdir(trainDir)
        trainingDataMat = np.zeros(shape=(len(trainFileLists), 32*32))
        # labelsVect存储0～9对应的index位置
        # trainingMat存放的每个位置对应的图片向量
        for i in range(len(trainFileLists)):
            fileNameStr = trainFileLists[i]
            # 获取每张图片的类别
            classifier = int(fileNameStr.split('_')[0])
            labelsVect.append(classifier)
            # 将 32*32的矩阵->1*1024的矩阵
            trainingDataMat[i, :] = self.image2vector(os.path.join(trainDir, fileNameStr))
        
        # 2.导入测试数据
        testFileLists = os.listdir(testDir)
        errorCount = 0
        for i in range(len(testFileLists)):
            fileNameStr = testFileLists[i]
            classifier = int(fileNameStr.split('_')[0])
            testDataVect = self.image2vector(os.path.join(testDir, fileNameStr))
            classifierPredict = self.classify(testDataVect, trainingDataMat, labelsVect, 3)
            print("the classifier came back with: {}, the real answer is: {}".format(classifierPredict, classifier))
            if(classifierPredict != classifier):
                errorCount += 1
        print("the error number is {}".format(errorCount))
        print("the total error rate is {}".format(float(errorCount)/float(len(testFileLists))))
    
        return trainingDataMat



if __name__ == "__main__":
    pt = SEKNN()
    '''
    filePath = './datasets/datingTestSet2.txt'
    featureMat, labelVector = pt.getDatasFromFile(filePath)
    print("feature shape is {}, feature datas {}".format(featureMat.shape, featureMat[:10]))
    print("label shape is {}, label datas {}".format(len(labelVector), labelVector[:10]))
    # 绘制数据集的分布
    plt.scatter(featureMat[:, 0], featureMat[:, 1], 15.0*np.array(labelVector), 15.0*np.array(labelVector))
    plt.show()

    normalFeature, ranges, minVals = pt.autoNormal(featureMat)
    print(ranges, minVals)
    print("normalFeature shape is {}, normalFeature data {}".format(normalFeature.shape, normalFeature[:10]), 
        type(normalFeature))
    
    # 数据集划分
    start = time.clock()
    testRatio = 0.1
    testNums = int(len(normalFeature)*testRatio)
    print("test datas number is {}".format(testNums))
    errorCount = 0.0 
    for i in range(testNums):
        # 对数据进行测试
        classifierResult = pt.classify(normalFeature[i, :], 
                            normalFeature[testNums:, :], labelVector[testNums:], 3)
        print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, labelVector[i]))
        if(classifierResult != labelVector[i]):
            errorCount += 1
    print("the error number is {}".format(errorCount))
    print("the total error rate is {}".format(float(errorCount)/float(testNums)))
    print("run time is {}".format(time.clock() - start))

    # 利用模型进行训练
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inArr = np.array([percentTats, ffMiles, iceCream])
    classifierResult = pt.classify((inArr-minVals)/ranges, normalFeature, labelVector, 3)
    print("You will probably like this person {}".format(resultList[classifierResult-1]))
    '''

    # 手写数字体识别
    trainDir = './datasets/trainingDigits'
    testDir = './datasets/testDigits'
    trianMat = pt.handWritingClassifier(trainDir, testDir)
    print(trianMat[0, :100])
