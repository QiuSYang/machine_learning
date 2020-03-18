"""
AdaBoost (adaptive boosting: 自适应 boosting) 概述
    能否使用弱分类器和多个实例来构建一个强分类器？ 这是一个非常有趣的理论问题。

AdaBoost 原理
AdaBoost 工作原理
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.AdaBoost/adaboost_illustration.png

AdaBoost 开发流程
    收集数据：可以使用任意方法
    准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。
        当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。
        作为弱分类器，简单分类器的效果更好。
    分析数据：可以使用任意方法。
    训练算法：AdaBoost 的大部分时间都用在训练上，分类器将多次在同一数据集上训练弱分类器。
    测试算法：计算分类的错误率。
    使用算法：通SVM一样，AdaBoost 预测两个类别中的一个。如果想把它应用到多个类别的场景，
                那么就要像多类 SVM 中的做法一样对 AdaBoost 进行修改。

AdaBoost 算法特点
    * 优点：泛化（由具体的、个别的扩大为一般的）错误率低，易编码，可以应用在大部分分类器上，无参数调节。
    * 缺点：对离群点敏感。
    * 适用数据类型：数值型和标称型数据。

项目案例: 马疝病的预测
项目流程图
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.AdaBoost/adaboost_code-flow-chart.jpg

基于单层决策树构建弱分类器
    单层决策树(decision stump, 也称决策树桩)是一种简单的决策树。

项目概述
    预测患有疝气病的马的存活问题，这里的数据包括368个样本和28个特征，疝气病是描述马胃肠痛的术语，
    然而，这种病并不一定源自马的胃肠问题，其他问题也可能引发疝气病，该数据集中包含了医院检测马疝气病的一些指标，有的指标比较主观，
    有的指标难以测量，例如马的疼痛级别。另外，除了部分指标主观和难以测量之外，该数据还存在一个问题，数据集中有30%的值是缺失的。

开发流程
    收集数据：提供的文本文件
    准备数据：确保类别标签是+1和-1，而非1和0
    分析数据：统计分析
    训练算法：在数据上，利用 adaBoostTrainDS() 函数训练出一系列的分类器
    测试算法：我们拥有两个数据集。在不采用随机抽样的方法下，我们就会对 AdaBoost 和 Logistic 回归的结果进行完全对等的比较
    使用算法：观察该例子上的错误率。不过，也可以构建一个 Web 网站，让驯马师输入马的症状然后预测马是否会死去

过拟合(overfitting, 也称为过学习)
    发现测试错误率在达到一个最小值之后有开始上升，这种现象称为过拟合。
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.AdaBoost/%E8%BF%87%E6%8B%9F%E5%90%88.png
    通俗来说：就是把一些噪音数据也拟合进去的，如下图。
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.AdaBoost/%E8%BF%87%E6%8B%9F%E5%90%88%E5%9B%BE%E8%A7%A3.png
"""
import os 
import sys 
import time 
import random 
import numpy as np 
import matplotlib.pyplot as plt 

class SEAdaBoost(object):
    def __init__(self):
        pass 
    
    def loadSimpData(self):
        """ 测试数据
        Returns:
            dataArr   feature对应的数据集
            labelArr  feature对应的分类标签
        """
        dataArr = np.array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
        labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
        return dataArr, labelArr
    
    # general function to parse tab-delimited floats
    def loadDataSet(self, fileName):
        # 获取feature的数量，便于获取
        #numFeat = len(open(fileName).readline.split('\t'))
        dataArr = []
        labelArr = []
        fr = open(fileName)
        if not fr:
            raise ValueError("open the {} file failure.".format(fileName))
        
        featureNums = 0
        index = 0
        for line in fr.readlines():
            lineArr = []
            currentLine = line.strip().split('\t')
            if index == 0:
                featureNums = len(currentLine)
            for i in range(featureNums-1):
                lineArr.append(float(currentLine[i]))
            dataArr.append(lineArr)
            labelArr.append(float(currentLine[-1]))
            index += 1
        
        print("the data size is {}".format(index))

        return dataArr, labelArr
    
    # 训练算法：在数据上，利用 adaBoostTrainDS() 函数训练出一系列的分类器
    '''
    发现：
        alpha （模型权重）目的主要是计算每一个分类器实例的权重(加和就是分类结果)
        分类的权重值：最大的值= alpha 的加和，最小值=-最大值
        D （样本权重）的目的是为了计算错误概率： weightedError = D.T*errArr，求最佳分类器
        样本的权重值：如果一个值误判的几率越小，那么 D 的样本权重越小

        https://github.com/apachecn/AiLearning/blob/master/img/ml/7.AdaBoost/adaboost_alpha.png
    '''
    def adaBoostTrainDS(self, dataArr, labelArr, numIt=40):
        """adaBoostTrainDS(adaBoost训练过程放大)
        Args:
            dataArr   特征标签集合
            labelArr  分类标签集合
            numIt     实例数，弱分类器数目
        Returns:
            weakClassArr  弱分类器的集合
            aggClassEst   预测的分类结果值
        """
        weakClassArr = []
        m = np.shape(dataArr)[0]
        # 初始化D, 设置每个样本的权重值, 平均分为m份
        # 一开始每个样本的权重一样
        D = np.mat(np.ones(shape=(m, 1))/m)
        aggClassEst = np.mat(np.zeros(shape=(m, 1)))
        for i in range(numIt):
            # 得到决策树的模型
            # classEst-训练分类结果
            bestStump, error, classEst = self._buildStump(dataArr, labelArr, D)
            
            # alpha目的主要是计算每一个分类器实例的权重(组合就是分类结果)
            # 计算每个分类器的alpha权重值
            alpha = float(0.5*np.log((1.0-error)/max(error, 1e-6)))
            bestStump['alpha'] = alpha 
            # store stump params in array
            weakClassArr.append(bestStump)

            print("alpha=%s, classEst=%s, bestStump=%s, error=%s " % (alpha, classEst.T, bestStump, error))
            # 分类正确：乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
            # 分类错误：乘积为 -1，结果会受影响，所以也乘以 -1
            expon = np.multiply(-1*alpha*np.mat(labelArr).T, classEst)
            print('(-1取反)预测值expon=', expon.T)
            # 计算e的expon次方，然后计算得到一个综合的概率的值
            # 结果发现： 判断错误的样本，D中相对应的样本权重值会变大
            D = np.multiply(D, np.exp(expon))
            D = D/D.sum()

            # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
            print('当前的分类结果：', alpha*classEst.T)
            aggClassEst += alpha*classEst
            print("叠加后的分类结果aggClassEst: ", aggClassEst.T)
            # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
            # 结果为：错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
            aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(labelArr).T, np.ones(shape=(m, 1)))
            errorRate = aggErrors.sum()/m
            #print("total error=%s" % (errorRate))
            if errorRate == 0.0:
                break
        
        return weakClassArr, aggClassEst
    
    # 测试算法：我们拥有两个数据集。在不采用随机抽样的方法下，
    # 我们就会对 AdaBoost 和 Logistic 回归的结果进行完全对等的比较。
    def adaClassify(self, datToClass, classifierArr):
        # do stuff similar to last aggClassEst in adaBoostTrainDs 
        dataMat = np.mat(datToClass)
        m = np.shape(dataMat)[0]
        aggClassEst = np.mat(np.zeros(shape=(m, 1)))

        # 循环多个分类器
        for i in range(len(classifierArr)):
            # 前提： 我们已经知道了最佳的分类器的实例
            # 通过分类器来核算每一次的分类结果，然后通过alpha*每一次的结果 得到最后的权重加和的值
            classEst = self._stumpClassify(dataMat, classifierArr[i]['dim'], 
                                        classifierArr[i]['thresh'], classifierArr[i]['ineq'])
            aggClassEst += classifierArr[i]['alpha']*classEst
            print(aggClassEst)
        
        return np.sign(aggClassEst)

    
    def plotROC(self, predStrengths, classLabels):
        """plotROC(打印ROC曲线，并计算AUC的面积大小)
        Args:
            predStrengths  最终预测结果的权重值
            classLabels    原始数据的分类结果集
        """
        print('predStrengths=', predStrengths)
        print('classLabels=', classLabels)
        # variable to calculate AUC 
        ySum = 0.0 
        # 对正样本的进行求和
        numPosClas = sum(np.array(classLabels)==1.0)
        # 对正样本的概率
        yStep = 1/float(numPosClas)
        # 负样本的概率
        xStep = 1/float(len(classLabels) - numPosClas)
        # argsort函数返回的是数组值从小到大的索引值
        # get sorted index, it's reverse
        sortedIndicies = predStrengths.argsort()
        # 测试结果是否是从小到大排列
        print('sortedIndicies=', sortedIndicies, predStrengths[0, 176], 
                predStrengths.min(), predStrengths[0, 293], predStrengths.max())
        
        # 开始创建模板对象
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        # cursor光标值
        cur = (1.0, 1.0)
        # loop through all the values, drawing a line segment at each point 
        for index in sortedIndicies.tolist()[0]:
            if classLabels[index] == 1.0:
                delX = 0
                delY = yStep 
            else:
                delX = xStep
                delY = 0
                ySum += cur[1]
            # draw line from cur to (cur[0]-delX, cur[1]-delY)
            # 画点连线 (x1, x2, y1, y2)
            print(cur[0], cur[0]-delX, cur[1], cur[1]-delY)
            ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
            cur = (cur[0]-delX, cur[1]-delY)
        # 画对角的虚线线
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        # 设置画图的范围区间 (x1, x2, y1, y2)
        ax.axis([0, 1, 0, 1])
        plt.show()
        '''
        参考说明：http://blog.csdn.net/wenyusuran/article/details/39056013
        为了计算 AUC ，我们需要对多个小矩形的面积进行累加。
        这些小矩形的宽度是xStep，因此可以先对所有矩形的高度进行累加，最后再乘以xStep得到其总面积。
        所有高度的和(ySum)随着x轴的每次移动而渐次增加。
        '''
        print("the Area Under the Curve is: ", ySum*xStep)


    def _buildStump(self, dataArr, labelArr, D):
        """buildStump(得到决策树的模型)
        Args:
            dataArr   特征标签集合
            labelArr  分类标签集合
            D         最初的样本的所有特征权重集合
        Returns:
            bestStump    最优的分类器模型
            minError     错误率
            bestClasEst  训练后的结果集
        """
        # 转换数据
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).T
        # m行 n行
        m, n = np.shape(dataMat)

        # 初始化数据
        numSteps = 10.0 
        bestStump = {}
        bestClasEst = np.mat(np.zeros(shape=(m, 1)))
        # 初始化的最小误差为无穷大
        minError = np.inf

        # 循环所有的feature列，将列切成若干份，每一段以最左边的点作为分类节点
        # n is feature number
        for i in range(n):
            rangeMin = dataMat[:, i].min()
            rangeMax = dataMat[:, i].max()
            #print("rangeMin = {}, rangeMax = {}".format(rangeMin, rangeMax))
            # 计算每一份的元素个数
            stepSize = (rangeMax - rangeMin)/numSteps
            # 例如： 4=(10-1)/2   那么  1-4(-1次)   1(0次)  1+1*4(1次)   1+2*4(2次)
            # 所以： 循环 -1/0/1/2
            for j in range(-1, int(numSteps)+1):
                # go over less than and greater than 
                for inequal in ['lt', 'gt']:
                    # 如果是-1，那么得到rangeMin-stepSize; 如果是numSteps，那么得到rangeMax
                    threshVal = (rangeMin + float(j) * stepSize)
                    # 对单层决策树进行简单分类，得到预测的分类值
                    # 根据阈值分类
                    predictedVals = self._stumpClassify(dataMat, i, threshVal, inequal)
                    # 计算错误率
                    errArr = np.mat(np.ones(shape=(m, 1)))
                    # 正确为0， 错误为1
                    errArr[predictedVals == labelMat] = 0
                    # 计算 平均每个特征的概率0.2*错误概率的总和为多少，就知道错误率多高
                    # 例如： 一个都没错，那么错误率= 0.2*0=0 ， 5个都错，那么错误率= 0.2*5=1， 
                    # 只错3个，那么错误率= 0.2*3=0.6
                    weightedError = D.T * errArr
                    '''
                    dim            表示 feature列
                    threshVal      表示树的分界值
                    inequal        表示计算树左右颠倒的错误率的情况
                    weightedError  表示整体结果的错误率
                    bestClasEst    预测的最优结果
                    '''
                    #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % 
                    #       (i, threshVal, inequal, weightedError))
                    if weightedError < minError:
                        minError = weightedError 
                        bestClasEst = predictedVals.copy()
                        bestStump['dim'] = i
                        bestStump['thresh'] = threshVal
                        bestStump['ineq'] = inequal
        
        # bestStump 表示分类器的结果，在第几个列上，用大于／小于比较，阈值是多少
        return bestStump, minError, bestClasEst

    
    def _stumpClassify(self, dataMat, dimen, threshVal, threshIneq):
        """stumpClassify(将数据集，按照feature列的value进行 二分法切分比较来赋值分类)
        Args:
            dataMat    Matrix数据集
            dimen      特征列
            threshVal  特征列要比较的值
        Returns:
            retArray 结果集
        """
        # 默认都是1
        retArray = np.ones(shape=(np.shape(dataMat)[0], 1))
        # dataMat[:, dimen] 表示数据集中第dimen列的所有值
        # threshIneq == 'lt'表示修改左边的值，gt表示修改右边的值
        # print('-----', threshIneq, dataMat[:, dimen], threshVal)
        if threshIneq == 'lt':
            retArray[dataMat[:, dimen] <= threshVal] = -1.0 
        else:
            retArray[dataMat[:, dimen] > threshVal] = -1.0
        
        return retArray


if __name__ == "__main__":
    pt = SEAdaBoost()
    # # 我们要将5个点进行分类
    # dataArr, labelArr = pt.loadSimpData()
    # print('dataArr', dataArr, 'labelArr', labelArr)

    # # D表示最初值，对1进行均分为5份，平均每一个初始的概率都为0.2
    # # D的目的是为了计算错误概率： weightedError = D.T*errArr
    # D = np.mat(np.ones((5, 1))/5)
    # print('D=', D.T)

    # # bestStump, minError, bestClasEst = pt.buildStump(dataArr, labelArr, D)
    # # print('bestStump=', bestStump)
    # # print('minError=', minError)
    # # print('bestClasEst=', bestClasEst.T)

    # # 分类器：weakClassArr
    # # 历史累计的分类结果集
    # weakClassArr, aggClassEst = pt.adaBoostTrainDS(dataArr, labelArr, 9)
    # print('\nweakClassArr=', weakClassArr, '\naggClassEst=', aggClassEst.T)

    # """
    # 发现:
    # 分类的权重值：最大的值，为alpha的加和，最小值为-最大值
    # 特征的权重值：如果一个值误判的几率越小，那么D的特征权重越少
    # """

    # # 测试数据的分类结果, 观测：aggClassEst分类的最终权重
    # print adaClassify([0, 0], weakClassArr).T
    # print adaClassify([[5, 5], [0, 0]], weakClassArr).T

    # 马疝病数据集
    # 训练集合
    dataArr, labelArr = pt.loadDataSet("./datasets/horseColicTraining2.txt")
    weakClassArr, aggClassEst = pt.adaBoostTrainDS(dataArr, labelArr, 40)
    print(weakClassArr, '\n-----\n', aggClassEst.T)
    # 计算ROC下面的AUC的面积大小
    pt.plotROC(aggClassEst.T, labelArr)
    # 测试集合
    dataArrTest, labelArrTest = pt.loadDataSet("./datasets/horseColicTest2.txt")
    m = np.shape(dataArrTest)[0]
    predicting10 = pt.adaClassify(dataArrTest, weakClassArr)
    errArr = np.mat(np.ones((m, 1)))
    # 测试：计算总样本数，错误样本数，错误率
    print(m, errArr[predicting10 != np.mat(labelArrTest).T].sum(), errArr[predicting10 != np.mat(labelArrTest).T].sum()/m)

