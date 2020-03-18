"""
树回归 概述
我们本章介绍 CART(Classification And Regression Trees， 分类回归树) 的树构建算法。该算法既可以用于分类还可以用于回归。

树回归 场景
我们在第 8 章中介绍了线性回归的一些强大的方法，但这些方法创建的模型需要拟合所有的样本点（局部加权线性回归除外）。
当数据拥有众多特征并且特征之间关系十分复杂时，构建全局模型的想法就显得太难了，也略显笨拙。
而且，实际生活中很多问题都是非线性的，不可能使用全局线性模型来拟合任何数据。
一种可行的方法是将数据集切分成很多份易建模的数据，然后利用我们的线性回归技术来建模。
如果首次切分后仍然难以拟合线性模型就继续切分。在这种切分方式下，树回归和回归法就相当有用。

除了我们在 第3章 中介绍的 决策树算法，我们介绍一个新的叫做 CART(Classification And Regression Trees, 分类回归树) 的树构建算法。
该算法既可以用于分类还可以用于回归。

1、树回归 原理
1.1、树回归 原理概述
为成功构建以分段常数为叶节点的树，需要度量出数据的一致性。
第3章使用树进行分类，会在给定节点时计算数据的混乱度。那么如何计算连续型数值的混乱度呢？

在这里，计算连续型数值的混乱度是非常简单的。
首先计算所有数据的均值，然后计算每条数据的值到均值的差值。
为了对正负差值同等看待，一般使用绝对值或平方值来代替上述差值。

上述做法有点类似于前面介绍过的统计学中常用的方差计算。
唯一不同就是，方差是平方误差的均值(均方差)，而这里需要的是平方误差的总值(总方差)。
总方差可以通过均方差乘以数据集中样本点的个数来得到。

1.2、树构建算法 比较
我们在 第3章 中使用的树构建算法是 ID3 。ID3 的做法是每次选取当前最佳的特征来分割数据，并按照该特征的所有可能取值来切分。
也就是说，如果一个特征有 4 种取值，那么数据将被切分成 4 份。
一旦按照某特征切分后，该特征在之后的算法执行过程中将不会再起作用，所以有观点认为这种切分方式过于迅速。
另外一种方法是二元切分法，即每次把数据集切分成两份。
如果数据的某特征值等于切分所要求的值，那么这些数据就进入树的左子树，反之则进入树的右子树。

除了切分过于迅速外， ID3 算法还存在另一个问题，它不能直接处理连续型特征。
只有事先将连续型特征转换成离散型，才能在 ID3 算法中使用。
但这种转换过程会破坏连续型变量的内在性质。而使用二元切分法则易于对树构造过程进行调整以处理连续型特征。
具体的处理方法是: 如果特征值大于给定值就走左子树，否则就走右子树。
另外，二元切分法也节省了树的构建时间，但这点意义也不是特别大，因为这些树构建一般是离线完成，时间并非需要重点关注的因素。

CART 是十分著名且广泛记载的树构建算法，它使用二元切分来处理连续型变量。
对 CART 稍作修改就可以处理回归问题。第 3 章中使用香农熵来度量集合的无组织程度。
如果选用其他方法来代替香农熵，就可以使用树构建算法来完成回归。

回归树与分类树的思路类似，但是叶节点的数据类型不是离散型，而是连续型。

1.2.1、附加 各常见树构造算法的划分分支方式
还有一点要说明，构建决策树算法，常用到的是三个方法: ID3, C4.5, CART.
三种方法区别是划分树的分支的方式:

ID3 是信息增益分支
C4.5 是信息增益率分支
CART 做分类工作时，采用 GINI 值作为节点分裂的依据；回归时，采用样本的最小方差作为节点的分裂依据。
工程上总的来说:

CART 和 C4.5 之间主要差异在于分类结果上，CART 可以回归分析也可以分类，C4.5 只能做分类；C4.5 子节点是可以多分的，而 CART 是无数个二叉子节点；

以此拓展出以 CART 为基础的 “树群” Random forest ， 以 回归树 为基础的 “树群” GBDT 。

1.3、树回归 工作原理
1、找到数据集切分的最佳位置，函数 chooseBestSplit() 伪代码大致如下:

对每个特征:
    对每个特征值: 
        将数据集切分成两份（小于该特征值的数据样本放在左子树，否则放在右子树）
        计算切分的误差
        如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
返回最佳切分的特征和阈值
2、树构建算法，函数 createTree() 伪代码大致如下:

找到最佳的待切分特征:
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在右子树调用 createTree() 方法
    在左子树调用 createTree() 方法
1.4、树回归 开发流程
(1) 收集数据：采用任意方法收集数据。
(2) 准备数据：需要数值型数据，标称型数据应该映射成二值型数据。
(3) 分析数据：绘出数据的二维可视化显示结果，以字典方式生成树。
(4) 训练算法：大部分时间都花费在叶节点树模型的构建上。
(5) 测试算法：使用测试数据上的R^2值来分析模型的效果。
(6) 使用算法：使用训练处的树做预测，预测结果还可以用来做很多事情。
1.5、树回归 算法特点
优点：可以对复杂和非线性的数据建模。
缺点：结果不易理解。
适用数据类型：数值型和标称型数据。
1.6、回归树 项目案例
1.6.1、项目概述
在简单数据集上生成一棵回归树。

1.6.2、开发流程
收集数据：采用任意方法收集数据
准备数据：需要数值型数据，标称型数据应该映射成二值型数据
分析数据：绘出数据的二维可视化显示结果，以字典方式生成树
训练算法：大部分时间都花费在叶节点树模型的构建上
测试算法：使用测试数据上的R^2值来分析模型的效果
使用算法：使用训练出的树做预测，预测结果还可以用来做很多事情
"""
import os 
import sys 
import random 
import time 
import numpy as np 
import matplotlib.pyplot as plt 

# 返回每一个叶子结点的均值
# returns the value used for each leaf
# 我的理解是：regLeaf 是产生叶节点的函数，就是求均值，即用聚类中心点来代表这类数据
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


# 计算总方差=方差*样本数
# 我的理解是：求这组数据的方差，即通过决策树划分，可以让靠近的数据分到同一类中去
def regErr(dataSet):
    # shape(dataSet)[0] 表示行数
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


 # 回归树测试案例
# 为了和 modelTreeEval() 保持一致，保留两个输入参数
def regTreeEval(model, inDat):
    """
    Desc:
        对 回归树 进行预测
    Args:
        model -- 指定模型，可选值为 回归树模型 或者 模型树模型，这里为回归树
        inDat -- 输入的测试数据
    Returns:
        float(model) -- 将输入的模型数据转换为 浮点数 返回
    """
    return float(model)


# 模型树测试案例
# 对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1，
# 也就是增加偏移值，和我们之前的简单线性回归是一个套路，增加一个偏移量
def modelTreeEval(model, inDat):
    """
    Desc:
        对 模型树 进行预测
    Args:
        model -- 输入模型，可选值为 回归树模型 或者 模型树模型，这里为模型树模型
        inDat -- 输入的测试数据
    Returns:
        float(X * model) -- 将测试数据乘以 回归系数 得到一个预测值 ，转化为 浮点数 返回
    """
    n = np.shape(inDat)[1]
    X = np.mat(np.ones(shape=(1, n+1)))
    X[:, 1: n+1] = inDat
    # print(X, model)
    return float(X * model)

class SERegTree(object):
    def __init__(self):
        pass 
    
    # 默认解析的数据是用tab分隔，并且是数值类型
    # general function to parse tab-delimited floats
    def loadDataSet(self, fileName):
        """loadDataSet(解析每一行，并转化为float类型)
            Desc：该函数读取一个以 tab 键为分隔符的文件，然后将每行的内容保存成一组浮点数
        Args:
            fileName 文件名
        Returns:
            dataMat 每一行的数据集array类型
        Raises:
        """
        # 假定最后一列是结果值
        # assume last column is target value
        dataMat = [] 
        fr = open(fileName)
        if not fr:
            raise ValueError("open the {} file is Failure.".format(fileName))
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            #print(curLine)
            # 将所有的元素转化为float类型
            # map all elements to float()
            # map() 函数具体的含义，可见 https://my.oschina.net/zyzzy/blog/115096
            #fltLine = map(float, curLine)
            fltLine = np.array(curLine, dtype=np.float).tolist()
            #print(fltLine)
            dataMat.append(fltLine)

        return dataMat

    def binSplitDataSet(self, dataSet, feature, value):
        """binSplitDataSet(将数据集，按照feature列的value进行 二元切分)
            Description：在给定特征和特征值的情况下，该函数通过数组过滤方式将上述数据集合切分得到两个子集并返回。
        Args:
            dataMat 数据集
            feature 待切分的特征列
            value 特征列要比较的值
        Returns:
            mat0 小于等于 value 的数据集在左边
            mat1 大于 value 的数据集在右边
        Raises:
        """
        # # 测试案例
        # print 'dataSet[:, feature]=', dataSet[:, feature]
        # print 'nonzero(dataSet[:, feature] > value)[0]=', nonzero(dataSet[:, feature] > value)[0]
        # print 'nonzero(dataSet[:, feature] <= value)[0]=', nonzero(dataSet[:, feature] <= value)[0]

        # dataSet[:, feature] 取去每一行中，第1列的值(从0开始算)
        # nonzero(dataSet[:, feature] > value)  返回结果为true行的index下标
        mat0 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :] # left subtree
        mat1 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]  # right subtree

        return mat0, mat1

    # 1.用最佳方式切分数据集
    # 2.生成相应的叶节点
    def chooseBestSplit(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        """chooseBestSplit(用最佳方式切分数据集 和 生成相应的叶节点)

        Args:
            dataSet   加载的原始数据集
            leafType  建立叶子点的函数
            errType   误差计算函数(求总方差)
            ops       [容许误差下降值，切分的最少样本数]。
        Returns:
            bestIndex feature的index坐标
            bestValue 切分的最优值
        Raises:
        """

        # ops=(1,4)，非常重要，因为它决定了决策树划分停止的threshold值，被称为预剪枝（prepruning），其实也就是用于控制函数的停止时机。
        # 之所以这样说，是因为它防止决策树的过拟合，所以当误差的下降值小于tolS，或划分后的集合size小于tolN时，选择停止继续划分。
        # 最小误差下降值，划分后的误差减小小于这个差值，就不用继续划分
        tolS = ops[0]
        # 划分最小size小于阈值就不继续划分
        tolN = ops[1]
        # 如果结果集(最后一列为1个变量)，就返回退出
        # .T 对数据集进行转置
        # .tolist()[0]转化为数组并取第0列
        # 如果集合size为1，不用继续划分。
        if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
            return None, leafType(dataSet)
        
        # 计算行列值
        m, n = np.shape(dataSet)
        # 无分类误差的总方差和
        # the choice of the best feature is driven by reduction in RSS error from mean 
        S = errType(dataSet)
        # inf 正无穷
        bestS, bestIndex, bestValue = np.inf, 0, 0
        # 循环处理每一列对应的feature值
        for featIndex in range(n-1):
            # 对于每个特征
            # [0]表示这一列的[所有行]，不要[0]就是一个array[[所有行]]
            for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
                # 对该列进行分组，然后组内的成员的val值进行 二元切分
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                # 判断二元切分的方式的元素数量是否符合预期, 判断数据量的大小
                if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                    continue 
                newS = errType(mat0) + errType(mat1) 
                # 如果二元切分，算出来的误差在可接受范围内，那么就记录切分点，并记录最小误差
                # 如果划分后误差小于bestS，则说明找到了新的bestS
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS

        # 判断二元切分的方式的元素误差是否符合预期
        # if the decrease (S-bestS) is less than a threshold don't do the split 
        if (S - bestS) < tolS:
            return None, leafType(dataSet)
        
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        # 对整体的成员进行判断，是否符合预期
        # 如果集合的size小于tolN
        if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
            # 当最佳划分后，集合过小，也不划分，产生叶节点
            return None, leafType(dataSet)
        
        return bestIndex, bestValue

    # assume dataSet is NumPy Mat so we can array filtering
    # 假设 dataSet 是 NumPy Mat 类型的，那么我们可以进行 array 过滤
    def createTree(self, dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
        """createTree(获取回归树)
            Description：递归函数：如果构建的是回归树，该模型是一个常数，如果是模型树，其模型师一个线性方程。
        Args:
            dataSet      加载的原始数据集, np.mat类型
            leafType     建立叶子点的函数
            errType      误差计算函数
            ops=(1, 4)   [容许误差下降值，切分的最少样本数]
        Returns:
            retTree    决策树最后的结果
        """
        # 选择最好的切分方式： feature索引值，最优切分值
        # choose the best split
        feat, val = self.chooseBestSplit(dataSet, leafType, errType, ops)
        # if the splitting hit a stop condition return val 
        # 如果 splitting 达到一个停止条件，就返回 val
        if feat is None:
            return val 
        
        retTree = {}
        retTree['spInd'] = feat 
        retTree['spVal'] = val 
        # 大于在右边，小于在左边，分为2个数据集
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        # 递归的进行调用，在左右子树中继续递归生成树
        retTree['left'] = self.createTree(lSet, leafType, errType, ops)
        retTree['right'] = self.createTree(rSet, leafType, errType, ops)

        return retTree 
    
    """
    2、树剪枝
    一棵树如果节点过多，表明该模型可能对数据进行了 “过拟合”。

    通过降低决策树的复杂度来避免过拟合的过程称为 剪枝（pruning）。在函数 chooseBestSplit() 中提前终止条件，
    实际上是在进行一种所谓的 预剪枝（prepruning）操作。另一个形式的剪枝需要使用测试集和训练集，称作 后剪枝（postpruning）。

    2.1、预剪枝(prepruning)
    顾名思义，预剪枝就是及早的停止树增长，在构造决策树的同时进行剪枝。

    所有决策树的构建方法，都是在无法进一步降低熵的情况下才会停止创建分支的过程，
    为了避免过拟合，可以设定一个阈值，熵减小的数量小于这个阈值，即使还可以继续降低熵，
    也停止继续创建分支。但是这种方法实际中的效果并不好。

    2.2、后剪枝(postpruning)
    决策树构造完成后进行剪枝。剪枝的过程是对拥有同样父节点的一组节点进行检查，
    判断如果将其合并，熵的增加量是否小于某一阈值。如果确实小，则这一组节点可以合并一个节点，
    其中包含了所有可能的结果。合并也被称作 塌陷处理 ，在回归树中一般采用取需要合并的所有子树的平均值。
    后剪枝是目前最普遍的做法。

    后剪枝 prune() 的伪代码如下:

    基于已有的树切分测试数据:
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并
    """
    # 判断节点是否是一个字典
    def isTree(self, obj):
        """
        Desc:
            测试输入变量是否是一棵树,即是否是一个字典
        Args:
            obj -- 输入变量
        Returns:
            返回布尔类型的结果。如果 obj 是一个字典，返回true，否则返回 false
        """
        return (type(obj).__name__ == 'dict')
    
    # 计算左右枝丫的均值
    def getMean(self, tree):
        """
        Desc:
            从上往下遍历树直到叶节点为止，如果找到两个叶节点则计算它们的平均值。
            对 tree 进行塌陷处理，即返回树平均值。
        Args:
            tree -- 输入的树
        Returns:
            返回 tree 节点的平均值
        """
        if self.isTree(tree['right']):
            tree['right'] = self.getMean(tree['right'])
        if self.isTree(tree['left']):
            tree['left'] = self.getMean(tree['left'])

        return (tree['left'] + tree['right'])/2.0
    
    # 检查是否适合合并分枝
    def prune(self, tree, testData):
        """
        Desc:
            从上而下找到叶节点，用测试数据集来判断将这些叶节点合并是否能降低测试误差
        Args:
            tree -- 待剪枝的树
            testData -- 剪枝所需要的测试数据 testData 
        Returns:
            tree -- 剪枝完成的树
        """
        # 判断是否测试数据集没有数据，如果没有，就直接返回tree本身的均值
        if np.shape(testData)[0] == 0:
            return self.getMean(tree)
        
        # 判断分支是否是dict字典，如果是就将测试数据集进行切分
        if self.isTree(tree['right']) or self.isTree(tree['left']):
            lSet, rSet = self.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 如果是左边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if self.isTree(tree['left']):
            tree['left'] = self.prune(tree['left'], lSet)
        # 如果是右边分枝是字典，就传入左边的数据集和左边的分枝，进行递归
        if self.isTree(tree['right']):
            tree['right'] = self.prune(tree['right'], rSet)
        
        # 上面的一系列操作本质上就是将测试数据集按照训练完成的树拆分好，对应的值放到对应的节点

        # 如果左右两边同时都不是dict字典，也就是左右两边都是叶节点，而不是子树了，那么分割测试数据集。
        # 1. 如果正确 
        #   * 那么计算一下总方差 和 该结果集的本身不分枝的总方差比较
        #   * 如果 合并的总方差 < 不合并的总方差，那么就进行合并
        # 注意返回的结果： 如果可以合并，原来的dict就变为了 数值
        if not self.isTree(tree['left']) and not self.isTree(tree['right']):
            lSet, rSet = self.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
            # power(x, y)表示x的y次方
            errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
            treeMean = (tree['left'] + tree['right'])/2.0
            errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
            # 如果合并的总方差 < 不合并的总方差，那么就进行合并
            if errorMerge < errorNoMerge:
                print('merging')
                return treeMean
            else:
                return tree 
        else:
            return tree 
        
    
    """
    3、模型树
    3.1、模型树 简介
    用树来对数据建模，除了把叶节点简单地设定为常数值之外，还有一种方法是把叶节点设定为分段线性函数，
    这里所谓的 分段线性（piecewise linear） 是指模型由多个线性片段组成。
    我们看一下图 9-4 中的数据，如果使用两条直线拟合是否比使用一组常数来建模好呢？答案显而易见。
    可以设计两条分别从 0.00.3、从 0.31.0 的直线，于是就可以得到两个线性模型。
    因为数据集里的一部分数据（0.00.3）以某个线性模型建模，而另一部分数据（0.31.0）则以另一个线性模型建模，
    因此我们说采用了所谓的分段线性模型。
    决策树相比于其他机器学习算法的优势之一在于结果更易理解。很显然，两条直线比很多节点组成一棵大树更容易解释。
    模型树的可解释性是它优于回归树的特点之一。另外，模型树也具有更高的预测准确度。
    https://github.com/apachecn/AiLearning/blob/master/img/ml/9.TreeRegression/RegTree_3.png

    分段线性数据
    将之前的回归树的代码稍作修改，就可以在叶节点生成线性模型而不是常数值。
    下面将利用树生成算法对数据进行划分，且每份切分数据都能很容易被线性模型所表示。这个算法的关键在于误差的计算。
    那么为了找到最佳切分，应该怎样计算误差呢？前面用于回归树的误差计算方法这里不能再用。
    稍加变化，对于给定的数据集，应该先用模型来对它进行拟合，然后计算真实的目标值与模型预测值间的差值。
    最后将这些差值的平方求和就得到了所需的误差。
    """

     # helper function used in two places
    def linearSolve(self, dataSet):
        """
        Desc:
            将数据集格式化成目标变量Y和自变量X，执行简单的线性回归，得到ws
        Args:
            dataSet -- 输入数据
        Returns:
            ws -- 执行线性回归的回归系数 
            X -- 格式化自变量X
            Y -- 格式化目标变量Y
        """
        m, n = np.shape(dataSet)
        # 产生一个关于1的矩阵
        X = np.mat(np.ones(shape=(m, n)))
        Y = np.mat(np.ones(shape=(m, 1)))
        # X的0列为1-常数项，用于计算平衡误差
        X[:, 1:n] = dataSet[:, 0:n-1]
        Y = dataSet[:, -1]

        # 转置矩阵*矩阵
        xTx = X.T * X 
        # 如果矩阵的逆不存在，会造成程序异常
        if np.linalg.det(xTx) == 0.0:
            raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
        # 最小二乘法求最优解:  w0*1+w1*x1=y
        # w = (X^T*X)^-1 * X^T *Y
        # xTx.I 求矩阵的逆
        ws = xTx.I * (X.T * Y)

        return ws, X, Y 
    
    # 得到模型的ws系数：f(x) = x0 + x1*featrue1+ x3*featrue2 ...
    # create linear model and return coeficients
    def modelLeaf(self, dataSet):
        """
        Desc:
            当数据不再需要切分的时候，生成叶节点的模型。
        Args:
            dataSet -- 输入数据集
        Returns:
            调用 linearSolve 函数，返回得到的 回归系数ws
        """
        ws, X, Y = self.linearSolve(dataSet)

        return ws
    
    # 计算线性模型的误差值
    def modelErr(self, dataSet):
        """
        Desc:
            在给定数据集上计算误差。
        Args:
            dataSet -- 输入数据集
        Returns:
            调用 linearSolve 函数，返回 yHat 和 Y 之间的平方误差。
        """
        ws, X, Y = self.linearSolve(dataSet)
        yHat = X * ws
        # print(corrcoef(yHat, Y, rowvar=0))
        return sum(np.power(Y - yHat, 2))
    
    """
    4、树回归 项目案例
    4.1、项目案例1: 树回归与标准回归的比较
    4.1.1、项目概述
    前面介绍了模型树、回归树和一般的回归方法，下面测试一下哪个模型最好。

    这些模型将在某个数据上进行测试，该数据涉及人的智力水平和自行车的速度的关系。当然，数据是假的。

    4.1.2、开发流程
        收集数据：采用任意方法收集数据
        准备数据：需要数值型数据，标称型数据应该映射成二值型数据
        分析数据：绘出数据的二维可视化显示结果，以字典方式生成树
        训练算法：模型树的构建
        测试算法：使用测试数据上的R^2值来分析模型的效果
        使用算法：使用训练出的树做预测，预测结果还可以用来做很多事情
    """
    # 计算预测的结果
    # 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
    # modelEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
    # 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
    # 调用modelEval()函数，该函数的默认值为regTreeEval()
    def treeForeCast(self, tree, inData, modelEval=regTreeEval):
        """
        Desc:
            对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        Args:
            tree -- 已经训练好的树的模型
            inData -- 输入的测试数据
            modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
        Returns:
            返回预测值
        """
        if not self.isTree(tree):
            return modelEval(tree, inData)
        if inData[tree['spInd']] <= tree['spVal']:
            if self.isTree(tree['left']):
                return self.treeForeCast(tree['left'], inData, modelEval)
            else:
                return modelEval(tree['left'], inData)
        else:
            if self.isTree(tree['right']):
                return self.treeForeCast(tree['right'], inData, modelEval)
            else:
                return modelEval(tree['right'], inData)
    
    # 预测结果
    def createForeCast(self, tree, testData, modelEval=regTreeEval):
        """
        Desc:
            调用 treeForeCast ，对特定模型的树进行预测，可以是 回归树 也可以是 模型树
        Args:
            tree -- 已经训练好的树的模型
            inData -- 输入的测试数据
            modelEval -- 预测的树的模型类型，可选值为 regTreeEval（回归树） 或 modelTreeEval（模型树），默认为回归树
        Returns:
            返回预测值矩阵
        """
        m = len(testData)
        print(m)
        yHat = np.mat(np.zeros(shape=(m, 1)))
        for i in range(m):
            print(np.mat(testData[i]))
            yHat[i, 0] = self.treeForeCast(tree, np.mat(testData[i]), modelEval)
        
        return yHat


"""
测试算法：使用测试数据上的R^2值来分析模型的效果

R^2 判定系数就是拟合优度判定系数，它体现了回归模型中自变量的变异在因变量的变异中所占的比例。
如 R^2=0.99999 表示在因变量 y 的变异中有 99.999% 是由于变量 x 引起。
当 R^2=1 时表示，所有观测点都落在拟合的直线或曲线上；当 R^2=0 时，表示自变量与因变量不存在直线或曲线关系。

所以我们看出， R^2 的值越接近 1.0 越好。

使用算法：使用训练出的树做预测，预测结果还可以用来做很多事情
"""
if __name__ == "__main__":
    pt = SERegTree()
    myDat = pt.loadDataSet('./datasets/data2.txt')
    print(myDat[0])
    myMat = np.mat(myDat)
    print('myMat=',  myMat[0])
    myTree = pt.createTree(myMat)
    print(myTree)

    trainMat = np.mat(pt.loadDataSet('./datasets/bikeSpeedVsIq_train.txt'))
    testMat = np.mat(pt.loadDataSet('./datasets/bikeSpeedVsIq_test.txt'))
    print(testMat[:, 0])
    # 回归树
    myTree1 = pt.createTree(trainMat, ops=(1, 20))
    print(myTree1)
    yHat1 = pt.createForeCast(myTree1, testMat[:, 0])
    print("--------------\n")
    print(yHat1)
    # # print "ssss==>", testMat[:, 1]
    # # corrcoef 返回皮尔森乘积矩相关系数
    print("regTree:", np.corrcoef(yHat1, testMat[:, 1],rowvar=0)[0, 1])

