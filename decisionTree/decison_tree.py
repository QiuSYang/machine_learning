"""
决策树概述：
决策树（Decision Tree）算法是一种基本的分类与回归方法，是最经常使用的数据挖掘算法之一。
我们这章节只讨论用于分类的决策树。
决策树模型呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。它可以认为是 if-then 规则的集合，
也可以认为是定义在特征空间与类空间上的条件概率分布。
决策树学习通常包括 3 个步骤：特征选择、决策树的生成和决策树的修剪。

决策树的定义：
1.分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点（node）和有向边（directed edge）组成。
结点有两种类型：内部结点（internal node）和叶结点（leaf node）。内部结点表示一个特征或属性(features)，叶结点表示一个类(labels)。
2.用决策树对需要测试的实例进行分类：从根节点开始，对实例的某一特征进行测试，根据测试结果，将实例分配到其子结点；
这时，每一个子结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最后将实例分配到叶结点的类中。

决策树原理：
决策树 须知概念
信息熵 & 信息增益
熵（entropy）： 熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。
信息论（information theory）中的熵（香农熵）： 是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。
例如：火柴有序放在火柴盒里，熵值很低，相反，熵值很高。
信息增益（information gain）： 在划分数据集前后信息发生的变化称为信息增益。

决策树 工作原理
如何构造一个决策树?
我们使用 createBranch() 方法，如下所示：
def createBranch():
'''
此处运用了迭代的思想。 感兴趣可以搜索 迭代 recursion， 甚至是 dynamic programing。
'''
    检测数据集中的所有数据的分类标签是否相同:
        If so return 类标签
        Else:
            寻找划分数据集的最好特征（划分之后信息熵最小，也就是信息增益最大的特征）
            划分数据集
            创建分支节点
                for 每个划分的子集
                    调用函数 createBranch （创建分支的函数）并增加返回结果到分支节点中
            return 分支节点

决策树 开发流程
收集数据：可以使用任何方法。
准备数据：树构造算法 (这里使用的是ID3算法，只适用于标称型数据，这就是为什么数值型数据必须离散化。 还有其他的树构造算法，比如CART)
分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
训练算法：构造树的数据结构。
测试算法：使用训练好的树计算错误率。
使用算法：此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义。

决策树 算法特点
优点：计算复杂度不高，输出结果易于理解，数据有缺失也能跑，可以处理不相关特征。
缺点：容易过拟合。
适用数据类型：数值型和标称型。

熵的计算公式：
    H = -求和(p(xi)log2 p(xi)), 其中p(xi)为选择该分类的概率
"""
import os 
import sys 
import time 
import math 
import operator 
import numpy as np 
import matplotlib.pyplot as plt 

'''
项目案例1: 判定鱼类和非鱼类
项目概述
根据以下 2 个特征，将动物分成两类：鱼类和非鱼类。

特征：
1. 不浮出水面是否可以生存
2. 是否有脚蹼
'''
class SEDecisonTree(object):
    def __init__(self):
        pass 
    
    def creatDatasets(self):
        dataSet = [[1, 1, 'yes'], 
                    [1, 1, 'yes'], 
                    [1, 0, 'no'], 
                    [0, 1, 'no'], 
                    [0, 1, 'no']]
        featureName = ['no surfacing', 'flippers']

        return dataSet, featureName
    
    # 计算给定数据集的香浓熵
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)
        # 计算分类标签label出现的次数
        labelsCounts = {}
        for theFeature in dataSet:
            # 获取当前数据的标签，数据的最后一列即为标签
            currentLabel = theFeature[-1]
            # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。
            # 每个键值都记录了当前类别出现的次数。
            if currentLabel not in labelsCounts.keys():
                labelsCounts[currentLabel] = 0
            labelsCounts[currentLabel] += 1
        # 对于label标签的占比，求出label标签的香浓熵
        shannonEnt = 0.0 
        for key in labelsCounts:
            # 使用所有类标签的发生频率计算类别出现的概率
            prob = float(labelsCounts[key])/float(numEntries)
            # 计算香农熵，以2为底求对数
            shannonEnt -= prob*math.log(prob, 2)
        
        return shannonEnt
    
    # 按照给定特征划分数据集
    def splitDataSet(self, dataSet, index, value):
        """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
            就是依据index列进行分类，如果index列的数据等于value的时候，
            就要将 index 划分到我们创建的新的数据集中
        Args:
            dataSet 数据集                 待划分的数据集
            index 表示每一行的index列        划分数据集的特征
            value 表示index列对应的value值   需要返回的特征的值。
        Returns:
            index列为value的数据集【该数据集需要排除index列】
        """
        retDataSet = []
        for theFeature in dataSet:
            # index列为value的数据集(该数据集需要排除index列)
            # 判断index列的值是否为value
            if theFeature[index] == value:
                reducedFeatVec = theFeature[:index]
                # 排除第index列的数据, 其实就是删除第index列数据， del theFeature[index]
                reducedFeatVec.extend(theFeature[index+1:])
                retDataSet.append(reducedFeatVec)

        return retDataSet

    # 选择最好的数据集划分方式，关键点
    '''
    问：上面的 newEntropy 为什么是根据子集计算的呢？
    答：因为我们在根据一个特征计算香农熵的时候，该特征的分类值是相同，这个特征这个分类的香农熵为 0；
    这就是为什么计算新的香农熵的时候使用的是子集。
    '''
    def chooseBestFeatureToSplit(self, dataSet):
        """chooseBestFeatureToSplit(选择最好的特征)

        Args:
            dataSet 数据集
        Returns:
            bestFeature 最优的特征列
        """
        # 求取特征数, -1因为最后一类为类别标签
        numFeatures = len(dataSet[0])-1
        # 计算数据集的原始
        baseEntropy = self.calcShannonEnt(dataSet)
        # 最优的信息增益值和最优的feature编号
        bestInfoGain, bestFeatureIndex = 0.0, -1
        # iterate over all the feature
        for i in range(numFeatures):
            # 获取第i个feature的所有数据
            theFeatureList = [example[i] for example in dataSet]
            # 剔除重复特征值
            uniqueVals = set(theFeatureList)
            # 创建临时的信息熵
            newEntropy = 0.0 
            # 遍历某一列的value集合，计算该列的信息熵 
            # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，
            # 计算数据集的新熵值，并对所有唯一特征值得到的熵求和
            for value in uniqueVals:
                # 在数据集选出第i列特征值为value，数据子集，并删除第i列的特征值
                subDataSet = self.splitDataSet(dataSet, i, value)
                # 计算子集占整个数据集的比例
                prob = float(len(subDataSet))/float(len(dataSet))
                # 计算子集信息熵的总和
                newEntropy += prob*self.calcShannonEnt(subDataSet)
            # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
            # 信息增益是熵的减少或者是数据无序度的减少。
            # 最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
            infoGain = baseEntropy - newEntropy
            #print("infoGain = {}, bestFeatureIndex = {}".format(infoGain, i))
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = i
        
        print("infoGain = {}, bestFeatureIndex = {}".format(bestInfoGain, bestFeatureIndex))
            
        return bestFeatureIndex

    def majorityCnt(self, classList):
        """majorityCnt(选择出现次数最多的一个结果)
        Args:
            classList label列的集合
        Returns:
            bestFeature 最优的特征列
        """
        # -----------majorityCnt的第一种方式 start------------------------------------
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        # print 'sortedClassCount:', sortedClassCount
        return sortedClassCount[0][0]
        # -----------majorityCnt的第一种方式 end------------------------------------

        # # -----------majorityCnt的第二种方式 start------------------------------------
        # major_label = Counter(classList).most_common(1)[0]
        # return major_label
        # # -----------majorityCnt的第二种方式 end------------------------------------

    def createTree(self, dataSet, features):
        print("datasets shape is {}".format(np.array(dataSet).shape))
        # 获取数据集的标签
        classList = [example[-1] for example in dataSet]

        # 下面两个if为判别函数
        # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，
        # 也就说只有一个类别，就只直接返回结果就行
        # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
        # count() 函数是统计括号中的值在list中出现的次数
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
        # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        
        # 选择最优的列，得到最优列对应的label含义
        bestFeatureIndex = self.chooseBestFeatureToSplit(dataSet)
        print("best feature name is '{}'".format(features[bestFeatureIndex]))
        # 获取label的名称
        bestFeatureName = features[bestFeatureIndex]
        # 初始化myTree
        myTree = {bestFeatureName: {}}
        # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
        # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
        del(features[bestFeatureIndex])
        # 取出最优列，然后它的branch做分类
        featureValues = [example[bestFeatureIndex] for example in dataSet]
        # 删除重复元素并排序
        uniqueVals = set(featureValues)
        for value in uniqueVals:
            # 求出剩余特征名称
            subFeatures = features[:]
            # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
            # 根据最优特征的值，将根节点链接几个子树
            myTree[bestFeatureName][value] = self.createTree(self.splitDataSet(dataSet, bestFeatureIndex, value), 
                                                            subFeatures)
            print("my tree is {}".format(myTree))
        
        return myTree

    def classify(self, inputTree, features, testVec):
        """classify(给输入的节点，进行分类)
        Args:
            inputTree  决策树模型
            featLabels Feature标签对应的名称
            testVec    测试输入的数据
        Returns:
            classLabel 分类的结果值，需要映射label才能知道名称
        """
        # 获取tree的根节点对于的key值
        firstStr = list(inputTree.keys())[0]
        # 通过key得到根节点对应的value
        secondDict = inputTree[firstStr]
        # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
        featIndex = features.index(firstStr)
        # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
        # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, features, testVec)
        else:
            classLabel = valueOfFeat
        return classLabel

    def storeTree(self, inputTree, filename):
        import pickle
        # -------------- 第一种方法 start --------------
        fw = open(filename, 'wb')
        pickle.dump(inputTree, fw)
        fw.close()
        # -------------- 第一种方法 end --------------

        # -------------- 第二种方法 start --------------
        with open(filename, 'wb') as fw:
            pickle.dump(inputTree, fw)
        # -------------- 第二种方法 start --------------
    
    def get_tree_height(self, tree):
        """
        Desc:
            递归获得决策树的高度
        Args:
            tree
        Returns:
            树高
        """

        if not isinstance(tree, dict):
            return 1

        child_trees = list(tree.values())[0].values()

        # 遍历子树, 获得子树的最大高度
        max_height = 0
        for child_tree in child_trees:
            child_tree_height = self.get_tree_height(child_tree)

            if child_tree_height > max_height:
                max_height = child_tree_height

        return max_height + 1


if __name__ == "__main__":
    import decisionTreePlot
    # 1.创建数据和结果标签
    pt = SEDecisonTree()
    myDat, labels = pt.creatDatasets()
    # print myDat, labels

    # 计算label分类标签的香农熵
    # calcShannonEnt(myDat)

    # # 求第0列 为 1/0的列的数据集【排除第0列】
    # print '1---', splitDataSet(myDat, 0, 1)
    # print '0---', splitDataSet(myDat, 0, 0)

    # # 计算最好的信息增益的列
    # print chooseBestFeatureToSplit(myDat)

    import copy
    myTree = pt.createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(pt.classify(myTree, labels, [1, 1]))
    
    # 获得树的高度
    print(pt.get_tree_height(myTree))

    # 画图可视化展现
    decisionTreePlot.createPlot(myTree)

    # 预测隐形眼镜的测试代码
    # 加载数据文件
    fr = open('./datasets/lenses.txt')
    if not fr:
        raise ValueError("open file is error.")
    # 解析数据，获取features数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 获取数据对应的feature名称
    lensesFeatureName = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 构建决策树模型
    lensesTree = pt.createTree(lenses, lensesFeatureName)
    print(lensesTree)
    # 树可视化
    decisionTreePlot.createPlot(lensesTree)


