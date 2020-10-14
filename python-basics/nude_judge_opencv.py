"""
# 使用传统图像处理算法实现色情图片鉴别, 图像处理使用python-OpenCV
参考链接：https://www.lanqiao.cn/courses/589/learning/?id=1964
原理：
本程序根据颜色（肤色）找出图片中皮肤的区域，然后通过一些条件判断是否为色情图片。

程序的关键步骤如下：
    1. 遍历每个像素，检测像素颜色是否为肤色
    2. 将相邻的肤色像素归为一个皮肤区域，得到若干个皮肤区域
    3. 剔除像素数量极少的皮肤区域

我们定义非色情图片的判定规则如下（满足任意一个判定为真）：
    1. 皮肤区域的个数小于 3 个
    2. 皮肤区域的像素与图像所有像素的比值小于 15%
    3. 最大皮肤区域小于总皮肤面积的 45%
    4. 皮肤区域数量超过 60 个
这些规则你可以尝试更改，直到程序效果让你满意为止。关于像素肤色判定这方面，公式可以在网上找到很多，
但世界上不可能有正确率 100% 的公式。你可以用自己找到的公式，在程序完成后慢慢调试。

RGB 颜色模式

第一种：r > 95 and g > 40 and g < 100 and b > 20 and max([r, g, b]) - min([r, g, b]) > 15 and abs(r - g) > 15 and r > g and r > b

第二种：nr = r / (r + g + b), ng = g / (r + g + b), nb = b / (r +g + b) ，nr / ng > 1.185 and r * b / (r + g + b) ** 2 > 0.107 and r * g / (r + g + b) ** 2 > 0.112

HSV 颜色模式

h > 0 and h < 35 and s > 0.23 and s < 0.68

YCbCr 颜色模式

97.5 <= cb <= 142.5 and 134 <= cr <= 176

一幅图像有零个到多个的皮肤区域，程序按发现顺序给它们编号，第一个发现的区域编号为 0，第 n 个发现的区域编号为 n-1。

我们用一种类型来表示像素，我们给这个类型取名为 Skin ，包含了像素的一些信息：
唯一的编号 id 、是/否肤色 skin 、皮肤区域号 region 、横坐标 x 、纵坐标 y 。

遍历所有像素时，我们为每个像素创建一个与之对应的 Skin 对象，并设置对象的所有属性。
其中 region 属性即为像素所在的皮肤区域编号，创建对象时初始化为无意义的 None 。
关于每个像素的 id 值，左上角为原点，像素的 id 值按像素坐标排布，那么看起来如下图：

2.2-1

其实 id 的顺序也即遍历的顺序。遍历所有像素时，创建 Skin 对象后，如果当前像素为肤色，
且相邻的像素有肤色的，那么我们把这些肤色像素归到一个皮肤区域。

相邻像素的定义：通常都能想到是当前像素周围的 8 个像素，然而实际上只需要定义 4 个就可以了，
位置分别在当前像素的左方，左上方，正上方，右上方；因为另外四个像素都在当前像素后面，
我们还未给这 4 个像素创建对应的 Skin 对象：

2.2-2
"""
import os 
import sys 
import cv2
import numpy as np 
from collections import namedtuple


class Nude(object):
    """色情图片鉴别"""
    Skin = namedtuple("Skin", "id skin region x y")
    def __init__(self, path_or_image):
        if isinstance(path_or_image, np.ndarray):
            self.image = path_or_image
        if isinstance(path_or_image, str):
            # 将图像转为彩色图像
            self.image = cv2.imread(path_or_image, flags=cv2.IMREAD_COLOR)

        if len(self.image.shape) == 2:
            # 将图像转为彩色图像
            self.image = np.expand_dims(self.image, axis=-1).repeat(3, axis=-1)
        
        # 存储对应图像所有像素的全部 Skin 对象
        self.skin_map = []
        # 检测到的皮肤区域，元素的索引即为皮肤区域号，元素都是包含一些 Skin 对象的列表
        self.detected_regions = []


if __name__ == "__main__":
    import argparse

    image_path = "./python-basics/1.jpg"
    pt = Nude(path_or_image=image_path)
