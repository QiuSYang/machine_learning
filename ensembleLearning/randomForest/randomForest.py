"""
随机森林
随机森林 概述
    随机森林指的是利用多棵树对样本进行训练并预测的一种分类器。
    决策树相当于一个大师，通过自己在数据集中学到的知识用于新数据的分类。
    但是俗话说得好，一个诸葛亮，玩不过三个臭皮匠。随机森林就是希望构建多个臭皮匠，希望最终的分类效果能够超过单个大师的一种算法。
随机森林 原理
那随机森林具体如何构建呢？
有两个方面：
    数据的随机性化
    待选特征的随机化
    使得随机森林中的决策树都能够彼此不同，提升系统的多样性，从而提升分类性能。

数据的随机化：使得随机森林中的决策树更普遍化一点，适合更多的场景。
（有放回的准确率在：70% 以上， 无放回的准确率在：60% 以上）
    1. 采取有放回的抽样方式 构造子数据集，保证不同子集之间的数量级一样（不同子集／同一子集 之间的元素可以重复）
    2. 利用子数据集来构建子决策树，将这个数据放到每个子决策树中，每个子决策树输出一个结果。
    3. 然后统计子决策树的投票结果，得到最终的分类 就是 随机森林的输出结果。
    4. 如下图，假设随机森林中有3棵子决策树，2棵子树的分类结果是A类，1棵子树的分类结果是B类，那么随机森林的分类结果就是A类。
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.RandomForest/%E6%95%B0%E6%8D%AE%E9%87%8D%E6%8A%BD%E6%A0%B7.jpg

待选特征的随机化
    1. 子树从所有的待选特征中随机选取一定的特征。
    2. 在选取的特征中选取最优的特征。
下图中，蓝色的方块代表所有可以被选择的特征，也就是目前的待选特征；黄色的方块是分裂特征。
左边是一棵决策树的特征选取过程，通过在待选特征中选取最优的分裂特征（别忘了前文提到的ID3算法，C4.5算法，CART算法等等），完成分裂。
右边是一个随机森林中的子树的特征选取过程。
    https://github.com/apachecn/AiLearning/blob/master/img/ml/7.RandomForest/%E7%89%B9%E5%BE%81%E9%87%8D%E6%8A%BD%E6%A0%B7.jpg


随机森林 开发流程：
收集数据：任何方法
准备数据：转换样本集
分析数据：任何方法
训练算法：通过数据随机化和特征随机化，进行多实例的分类评估
测试算法：计算错误率
使用算法：输入样本数据，然后运行 随机森林 算法判断输入数据分类属于哪个分类，最后对计算出的分类执行后续处理

随机森林 算法特点：
优点：几乎不需要输入准备、可实现隐式特征选择、训练速度非常快、其他模型很难超越、
    很难建立一个糟糕的随机森林模型、大量优秀、免费以及开源的实现。
缺点：劣势在于模型大小、是个很难去解释的黑盒子。
适用数据范围：数值型和标称型
"""
import os 
import sys 
import random 
import time 
import numpy as np 
import matplotlib.pyplot as plt 

'''
开发流程
收集数据：提供的文本文件
准备数据：转换样本集
分析数据：手工检查数据
训练算法：在数据上，利用 random_forest() 函数进行优化评估，返回模型的综合分类结果
测试算法：在采用自定义 n_folds 份随机重抽样 进行测试评估，得出综合的预测评分
使用算法：若你感兴趣可以构建完整的应用程序，从案例进行封装，也可以参考我们的代码
'''
class SERandomForest(object):
    def __init__(self):
        pass 
    
    # 导入csv文件
    def loadDataSet(self, filename):
        dataset = []
        with open(filename, 'r') as fr:
            for line in fr.readlines():
                if not line:
                    continue
                lineArr = []
                '''
                for feature in line.split(','):
                    # strip()返回移除字符串头尾指定的字符生成的新字符串
                    str_f = feature.strip()
                    # 判断是否是数字
                    if str_f.isdigit():
                        # 将数据集的第column列转换成float形式
                        lineArr.append(float(str_f))
                        
                    else:
                        # 添加分类标签
                        lineArr.append(str_f)
                '''
                lineContent = line.split(',')
                #print(len(lineContent))
                for i in range(len(lineContent)):
                    if i == len(lineContent)-1:
                        # 添加分类标签
                        lineArr.append(lineContent[i])
                    else:
                        # 将数据集的第column列转换成float形式
                        lineArr.append(float(lineContent[i]))
                        
                dataset.append(lineArr)
            
        return dataset

    '''
    分析数据：手工检查数据
    训练算法：在数据上，利用 random_forest() 函数进行优化评估，返回模型的综合分类结果
    '''
    #样本数据随机无放回抽样-用于交叉验证
    def cross_validation_split(self, dataset, n_folds):
        """cross_validation_split(将数据集进行抽重抽样 n_folds 份，数据可以重复抽取)

        Args:
            dataset     原始数据集
            n_folds     数据集dataset分成n_flods份
        Returns:
            dataset_split    list集合，存放的是：将数据集进行抽重抽样 n_folds 份，数据可以重复抽取
        """
        dataset_split = list()
        # 复制一份dataset, 防止dataset的内容改变
        #dataset_copy = list(dataset)
        dataset_copy = dataset.copy()
        fold_size = int(len(dataset) / n_folds)
        #print(fold_size)
        for i in range(n_folds):
            # 每次循环fold清零，防止重复导入dataset_split
            fold = list()
            # 这里不能用if, if只是在第一次判断时起作用，while执行循环，直到条件不成立
            while len(fold) < fold_size:
                # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，有的则从未在训练集中出现，
                # 此为自助采样法。从而保证每棵决策树训练集的差异性 
                index = random.randrange(len(dataset_copy))
                #count = len(dataset_copy)
                #index = random.randrange(count)
                #print(index)
                # 将对应索引 index 的内容从 dataset_copy 中导出，并将该内容从 dataset_copy 中删除。
                # pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
                # fold.append(dataset_copy.pop(index))  # 无放回的方式
                fold.append(dataset_copy.pop(index)) # 有放回的方式

            dataset_split.append(fold)

        # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证
        return dataset_split
    
    # Split a dataset based on an attribute and an attribute value 
    # 根据特征和特征值分割数据集
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        
        return left, right

    
    '''
    Gini指数的计算问题，假如将原始数据集D切割两部分，分别为D1和D2，则
    Gini(D|切割) = (|D1|/|D| ) * Gini(D1) + (|D2|/|D|) * Gini(D2)
    学习地址：
        http://bbs.pinggu.org/thread-5986969-1-1.html
        http://www.cnblogs.com/pinard/p/6053344.html
    而原文中 计算方式为：
    Gini(D|切割) = Gini(D1) + Gini(D2)
    # Calculate the Gini index for a split dataset
    def gini_index(groups, class_values):    # 个人理解：计算代价，分类越准确，则 gini 越小
        gini = 0.0
        for class_value in class_values:     # class_values = [0, 1] 
            for group in groups:             # groups = (left, right)
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += (proportion * (1.0 - proportion))    # 个人理解：计算代价，分类越准确，则 gini 越小
        return gini
    '''
    # 个人理解：计算代价，分类越准确，则gini越小
    def gini_index(self, groups, class_values):
        gini = 0.0 
        D = len(groups[0]) + len(groups[1])
        # class_values = [0, 1]
        for class_value in class_values:
            # groups = (left, right)
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += float(size)/D * (proportion * (1.0 - proportion))
        
        return gini
    

    # Create a random subsample from the dataset with replacement
    # 创建数据集的随机子样本
    def subsample(self, dataset, ratio):
        """random_forest(评估算法性能，返回模型得分)

        Args:
            dataset         训练数据集
            ratio           训练数据集的样本比例
        Returns:
            sample          随机抽样的训练样本
        """
        sample = list()
        # 训练样本按比例抽样
        # round() 方法返回浮点数x的四舍五入值
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，
            # 有的则从未在训练集中出现，此为自助采样法。从而保证每棵决策树训练集的差异性
            index = random.randrange(len(dataset))
            sample.append(dataset[index])
        
        return sample

    # 特征随机化
    # 找出分割数据集的最优特征，得到最优的特征 index，特征值 row[index]，
    # 以及分割完的数据 groups（left, right）
    def get_split(self, dataset, n_feature):
        # class_values = [0, 1]
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None 
        feature = list()
        while len(feature) < n_feature:
            # 往 features 添加 n_features 个特征（ n_feature 等于特征数的个数），特征索引从 dataset 中随机取
            index = random.randrange(len(dataset[0])-1)
            if index not in feature:
                feature.append(index)
        # 在 n_features 个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
        for index in feature:
            for row in dataset:
                # groups=(left, right), row[index] 遍历每一行 index 索引下的特征值作为分类值 value, 找出最优的分类特征和特征值
                #print(index, type(row[index]))
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
                if gini < b_score:
                    # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups, b_value 为分错的代价成本
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        
        return {'index': b_index, 'value': b_value, 'groups': b_groups}
    
    # Random Forest Algorithm 
    # 随机森林
    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        """random_forest(评估算法性能，返回模型得分)

        Args:
            train           训练数据集
            test            测试数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            sample_size     训练数据集的样本比例
            n_trees         决策树的个数
            n_features      选取的特征的个数
        Returns:
            predictions     每一行的预测结果，bagging 预测最后的分类结果
        """
        trees = list()
        # n_trees 表示决策树的数量
        for i in range(n_trees):
            # 随机抽样的训练样本，随机采样保证了每棵决策树训练集的差异性
            sample = self.subsample(train, sample_size)
            # 创建决策树
            tree = self._build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
        
        # 每个数据样本的预测结果，bagging预测最后的分类结果
        predictions = [self.bagging_predict(trees, row) for row in test]

        return predictions

    # Calculate accuracy percentage 
    def accuracy_metric(self, actual, predicted):
        # 导入实际值和预测值，计算精确度
        correct = 0 
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        
        return correct / float(len(actual)) * 100.0 
    
    # 评估算法性能， 返回模型得分
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        """evaluate_algorithm(评估算法性能，返回模型得分)
        Args:
            dataset     原始数据集
            algorithm   使用的算法
            n_folds     数据的份数
            *args       其他的参数
        Returns:
            scores      模型得分
        """

        # 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次 list 的元素是无重复的
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
       # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
             # 将多个 fold 列表组合成一个 train_set 列表, 类似 union all
            """
            In [20]: l1=[[1, 2, 'a'], [11, 22, 'b']]
            In [21]: l2=[[3, 4, 'c'], [33, 44, 'd']]
            In [22]: l=[]
            In [23]: l.append(l1)
            In [24]: l.append(l2)
            In [25]: l
            Out[25]: [[[1, 2, 'a'], [11, 22, 'b']], [[3, 4, 'c'], [33, 44, 'd']]]
            In [26]: sum(l, [])
            Out[26]: [[1, 2, 'a'], [11, 22, 'b'], [3, 4, 'c'], [33, 44, 'd']]
            """
            train_set = sum(train_set, [])
            test_set = list()
            # fold 表示从原始数据集 dataset 提取出来的测试集
            for row in fold:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]

            # 计算随机森林的预测结果的正确率
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        
        return scores

    
    # build a decision tree
    def _build_tree(self, train, max_depth, min_size, n_features):
        """build_tree(创建一个决策树)
        Args:
            train           训练数据集
            max_depth       决策树深度不能太深，不然容易导致过拟合
            min_size        叶子节点的大小
            n_features      选取的特征的个数
        Returns:
            root            返回决策树
        """
        # 返回最优列和相关的信息
        root = self.get_split(train, n_features)
        
        # 对左右2边的数据 进行递归的调用，由于最优特征使用过，
        # 所以在后面进行使用的时候，就没有意义了
        # 例如： 性别-男女，对男使用这一特征就没任何意义了
        self._split(root, max_depth, min_size, n_features, 1)

        return root

    
    # Make a prediction with a decision tree 
    # 预测模型分类结果
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            # isinstance 是 Python 中的一个内建函数。是用来判断一个对象是否是一个已知的类型。
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
    

    # Make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        """bagging_predict(bagging预测)
        Args:
            trees           决策树的集合
            row             测试数据集的每一行数据
        Returns:
            返回随机森林中，决策树结果出现次数做大的
        """

        # 使用多个决策树trees对测试集test的第row行进行预测，再使用简单投票法判断出该行所属分类
        predictions = [self.predict(trees, row) for trees in trees]
        
        return max(set(predictions), key=predictions.count)


    # Create a terminal node value 
    # 输出group中出现次数较多的标签
    def _to_terminal(self, group):
        # max()函数中，当key参数不为空时， 就以key的函数对象为判断标准
        outcomes = [row[-1] for row in group]
        # 输出group中出现次数较多的标签
        return max(set(outcomes), key=outcomes.count)
    
    # Create child splits for a node or make terminal 
    # 创建子分割器，递归分类，直到分类结束
    def _split(self, node, max_depth, min_size, n_feature, depth):
        # max_depth = 10, min_size = 1, n_features=int(sqrt((len(dataset[0])-1)
        left, right = node['groups']
        del node['groups']
        # check for a no split 
        if not left or not right:
            node['left'] = node['right'] = self._to_terminal(left + right)
            return 
        # check for max depth 
        if depth >= max_depth:
            # max_depth=10 表示递归十次，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
            node['left'], node['right'] = self._to_terminal(left), self._to_terminal(right)
            return 
        # process left child 
        if len(left) <= min_size:
            node['left'] = self._to_terminal(left)
        else:
            # node['left']是一个字典，形式为{'index':b_index, 'value':b_value, 'groups':b_groups}，所以node是一个多层字典
            node['left'] = self.get_split(left, n_feature)
            # 递归，depth+1计算递归层数
            self._split(node['left'], max_depth, min_size, n_feature, depth+1)
        # process right child 
        if len(right) <= min_size:
            node['right'] = self._to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_feature)
            self._split(node['right'], max_depth, min_size, n_feature, depth+1)


if __name__ == "__main__":
    # 加载数据
    pt = SERandomForest()
    dataset = pt.loadDataSet('./datasets/sonar-all-data.txt')
    print(dataset[0][0], type(dataset[0][0]))

    n_folds = 5        # 分成5份数据，进行交叉验证
    max_depth = 20     # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1       # 决策树的叶子节点最少的元素数量
    sample_size = 1.0  # 做决策树时候的样本的比例
    # n_features = int((len(dataset[0])-1))
    n_features = 15     # 调参（自己修改） #准确性与多样性之间的权衡
    for n_trees in [1, 10, 20, 30, 40, 50]:  # 理论上树是越多越好
        scores = pt.evaluate_algorithm(dataset, pt.random_forest, n_folds, 
                                    max_depth, min_size, sample_size, n_trees, n_features)
        # 每一次执行本文件时都能产生同一个随机数
        random.seed(1)
        print('random=', random.random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    

