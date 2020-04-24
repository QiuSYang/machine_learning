Histogram of Oriented Gridients(HOG) 方向梯度直方图
	Histogram of Oriented Gridients，缩写为HOG，是目前计算机视觉、模式识别领域很常用的一种描述图像局部纹理的特征。
	这个特征名字起的也很直白，就是说先计算图片某一区域中不同方向上梯度的值，然后进行累积，得到直方图，这个直方图呢，
	就可以代表这块区域了，也就是作为特征，可以输入到分类器里面了。
	那么，接下来介绍一下HOG的具体原理和计算方法，以及一些引申。

工程原理：
	1.使用hog图像分析提取图像特征；
	2.使用hog分析之后提取的图像特征作为SVM模型的输入数据；
	3.训练SVM分类器进行Image分类。

dataset website: https://www.kaggle.com/rhammell/ships-in-satellite-imagery
