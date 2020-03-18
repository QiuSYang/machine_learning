"""
使用朴素贝叶斯做邮件分类
"""
import os 
import sys 
import re 
import random
import numpy as np 
from bayes import SEBayes 

# 切分文本
def textParse(bigString):
    '''
    Desc:
        接收一个大字符串并将其解析为字符串列表
    Args:
        bigString -- 大字符串
    Returns:
        去掉少于 2 个字符的字符串，并将所有字符串转换为小写，返回字符串列表
    '''
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    '''
    Desc:
        对贝叶斯垃圾邮件分类器进行自动化处理。
    Args:
        none
    Returns:
        对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 切分，解析数据， 并归类为1类别
        wordList = textParse(open('./datasets/email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据， 并归类为1类别
        wordList = textParse(open('./datasets/email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    
    # 声明byes类对象
    byesPt = SEBayes()
    # 创建词汇表
    vocabList = byesPt.createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 随机选取10个邮件用来测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x ~ y 的实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(byesPt.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    # 模型训练
    p0V, p1V, pSpam = byesPt.trainNB0(np.array(trainMat), np.array(trainClasses))
    print(p0V, p1V, pSpam)
    # 模型测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = byesPt.setOfWords2Vec(vocabList, docList[docIndex])
        if byesPt.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    
    print('the error count is {}'.format(errorCount))
    print('the error rate is {}'.format(float(errorCount)/len(testSet)))

def testParseTest():
    print(textParse(open('./datasets/email/ham/{}.txt'.format(1)).read()))


if __name__ == "__main__":
    testParseTest()
    spamTest()


