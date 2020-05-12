"""
# 加载已经训练好的svm model, load model parameter for the function cv::HOGDescriptor::setSVMDetector() parameter

# C++ codes

# #include<opencv/cv.h>
# #include<opencv2/core/core.hpp>
# #include<opencv2/highgui/highgui.hpp>
# #include<opencv2/opencv.hpp>
# #include<opencv2/gpu/gpu.hpp>
# #include<opencv2/ml/ml.hpp>
# #include<opencv2/objdetect/objdetect.hpp>
# #include<iostream>
# #include<fstream>
# #include<string>
# #include<vector>
# using namespace std;
# using namespace cv;
#
#
# #define TRAIN //开关控制是否训练还是直接载入训练好的模型
#
# class MySVM: public CvSVM
# {
# public:
# 	double * get_alpha_data()
# 	{
# 		return this->decision_func->alpha;
# 	}
# 	double  get_rho_data()
# 	{
# 		return this->decision_func->rho;
# 	}
# };
#
# void main(int argc, char ** argv)
# {
#
# 	MySVM SVM;
# 	int descriptorDim;
#
# 	string buffer;
# 	string trainImg;
# 	vector<string> posSamples;
# 	vector<string> negSamples;
# 	vector<string> testSamples;
# 	int posSampleNum;
# 	int negSampleNum;
# 	int testSampleNum;
# 	string basePath = "";//相对路径之前加上基地址，如果训练样本中是相对地址，则都加上基地址
# 	double rho;
#
# #ifdef TRAIN
# 		ifstream fInPos("D:\\DataSet\\CarFaceDataSet\\PositiveSample.txt");//读取正样本
# 		ifstream fInNeg("D:\\DataSet\\CarFaceDataSet\\NegtiveSample.txt");//读取负样本
#
# 		while (fInPos)//讲正样本读入imgPathList中
# 		{
# 			if(getline(fInPos, buffer))
# 				posSamples.push_back(basePath + buffer);
# 		}
# 		posSampleNum = posSamples.size();
# 		fInPos.close();
#
# 		while(fInNeg)//读取负样本
# 		{
# 			if (getline(fInNeg, buffer))
# 				negSamples.push_back(basePath + buffer);
# 		}
# 		negSampleNum = negSamples.size();
# 		fInNeg.close();
#
# 		Mat sampleFeatureMat;//样本特征向量矩阵
# 		Mat sampleLabelMat;//样本标签
#
# 		HOGDescriptor * hog = new HOGDescriptor (cvSize(128, 128), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
# 		vector<float> descriptor;
#
# 		for(int i = 0 ; i < posSampleNum; i++)// 处理正样本
# 		{
# 			Mat inputImg = imread(posSamples[i]);
# 			cout<<"processing "<<i<<"/"<<posSampleNum<<" "<<posSamples[i]<<endl;
# 			Size dsize = Size(128,128);
# 			Mat trainImg = Mat(dsize, CV_32S);
# 			resize(inputImg, trainImg, dsize);
#
# 			hog->compute(trainImg, descriptor, Size(8, 8));
# 			descriptorDim = descriptor.size();
#
# 			if(i == 0)//首次特殊处理根据检测到的维数确定特征矩阵的尺寸
# 			{
# 				sampleFeatureMat = Mat::zeros(posSampleNum + negSampleNum, descriptorDim, CV_32FC1);
# 				sampleLabelMat = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32FC1);
# 			}
#
# 			for(int j = 0; j < descriptorDim; j++)//将特征向量复制到矩阵中
# 			{
# 				sampleFeatureMat.at<float>(i, j) = descriptor[j];
# 			}
#
# 			sampleLabelMat.at<float>(i, 0) = 1;
# 		}
#
# 		cout<<"extract posSampleFeature done"<<endl;
#
# 		for(int i = 0 ; i < negSampleNum; i++)//处理负样本
# 		{
# 			Mat inputImg = imread(negSamples[i]);
# 			cout<<"processing "<<i<<"/"<<negSampleNum<<" "<<negSamples[i]<<endl;
# 			Size dsize = Size(128,128);
# 			Mat trainImg = Mat(dsize, CV_32S);
# 			resize(inputImg, trainImg, dsize);
# 			hog->compute(trainImg, descriptor, Size(8,8));
# 			descriptorDim = descriptor.size();
#
# 			for(int j = 0; j < descriptorDim; j++)//将特征向量复制到矩阵中
# 			{
# 				sampleFeatureMat.at<float>(posSampleNum + i, j) = descriptor[j];
# 			}
#
# 			sampleLabelMat.at<float>(posSampleNum + i, 0) = -1;
# 		}
#
# 		cout<<"extract negSampleFeature done"<<endl;
#
# 		//此处先预留hard example 训练后再添加
#
# 		ofstream foutFeature("SampleFeatureMat.txt");//保存特征向量文件
# 		for(int i = 0; i <  posSampleNum + negSampleNum; i++)
# 		{
# 			for(int j = 0; j < descriptorDim; j++)
# 			{
# 				foutFeature<<sampleFeatureMat.at<float>(i, j)<<" ";
# 			}
# 			foutFeature<<"\n";
# 		}
# 		foutFeature.close();
# 		cout<<"output posSample and negSample Feature done"<<endl;
#
# 		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
# 	    CvSVMParams params(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);  //这里一定要注意，LINEAR代表的是线性核，RBF代表的是高斯核，如果要用opencv自带的detector必须用线性核，如果自己写，或者只是判断是否为车脸的2分类问题则可以用RBF，在此应用环境中线性核的性能还是不错的
#     	cout<<"SVM Training Start..."<<endl;
# 		SVM.train_auto(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), params);
# 		SVM.save("SVM_Model.xml");
# 		cout<<"SVM Training Complete"<<endl;
# #endif
#
# #ifndef TRAIN
# 		SVM.load("SVM_Model.xml");//加载模型文件
# #endif
#
# ------------------------------------------------------ #
# 线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha, 有一个浮点数，叫做rho;
# 将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
# 如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
# 就可以利用你的训练样本训练出来的分类器进行行人检测了。
# ----------------------------------------------------- #
# 	descriptorDim = SVM.get_var_count();  // 特征向量的维数，即HOG描述子的维数
# 	int supportVectorNum = SVM.get_support_vector_count();  # 支持向量的个数
# 	cout<<"support vector num: "<< supportVectorNum <<endl;
#
# 	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);  // alpha向量，长度等于支持向量个数
# 	Mat supportVectorMat = Mat::zeros(supportVectorNum, descriptorDim, CV_32FC1);  // 支持向量矩阵
# 	Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1);  // alpha向量乘以支持向量矩阵的结果
#
#   // 将支持向量的数据复制到supportVectorMat矩阵中
# 	for (int i = 0; i < supportVectorNum; i++)//复制支持向量矩阵
# 	{
#       //返回第i个支持向量的数据指针
# 		const float * pSupportVectorData = SVM.get_support_vector(i);
# 		for(int j = 0 ;j < descriptorDim; j++)
# 		{
# 			supportVectorMat.at<float>(i,j) = pSupportVectorData[j];
# 		}
# 	}
#
#   // alpha向量的数据复制到alphaMat中
# 	double *pAlphaData = SVM.get_alpha_data(); // 返回SVM的决策函数中的alpha向量
# 	for (int i = 0; i < supportVectorNum; i++) //复制函数中的alpha 记住决策公式Y= wx+b
# 	{
# 		alphaMat.at<float>(0, i) = pAlphaData[i];
# 	}
#
# 	resultMat = -1 * alphaMat * supportVectorMat; //alphaMat就是权重向量
#
# 	//cout<<resultMat;
#
# 	cout<<"描述子维数 "<<descriptorDim<<endl;
#   // 得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
# 	vector<float> myDetector;
# 	for (int i = 0 ;i < descriptorDim; i++)
# 	{
# 		myDetector.push_back(resultMat.at<float>(0, i));
# 	}
#
# 	rho = SVM.get_rho_data();
# 	myDetector.push_back(rho);
# 	cout<<"检测子维数 "<<myDetector.size()<<endl;
#
# 	HOGDescriptor myHOG (Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
# 	myHOG.setSVMDetector(myDetector);//设置检测子
#
# 	//保存检测子
# 	int minusNum = 0;
# 	int posNum = 0;
#
# 	ofstream foutDetector("HogDetectorForCarFace.txt");
# 	for (int i = 0 ;i < myDetector.size(); i++)
# 	{
# 		foutDetector<<myDetector[i]<<" ";
# 		//cout<<myDetector[i]<<" ";
# 	}
#
# 	//cout<<endl<<"posNum "<<posNum<<endl;
# 	//cout<<endl<<"minusNum "<<minusNum<<endl;
# 	foutDetector.close();
# 	//test part
# 	ifstream fInTest("D:\\DataSet\\CarFaceDataSet\\testSample.txt");
# 	while (fInTest)
# 	{
# 		if(getline(fInTest, buffer))
# 		{
# 			testSamples.push_back(basePath + buffer);
# 		}
# 	}
# 	testSampleNum = testSamples.size();
# 	fInTest.close();
#
# 	for (int i = 0; i < testSamples.size(); i++)
# 	{
# 		Mat testImg = imread(testSamples[i]);
# 		Size dsize = Size(320, 240);
# 		Mat testImgNorm (dsize, CV_32S);
# 		resize(testImg, testImgNorm, dsize);
#
# 		vector<Rect> found, foundFiltered;
# 		cout<<"MultiScale detect "<<endl;
# 		myHOG.detectMultiScale(testImgNorm, found, 0, Size(8,8), Size(0,0), 1.05, 2);
# 		cout<<"Detected Rect Num"<< found.size()<<endl;
#
# 		for (int i = 0; i < found.size(); i++)//查看是否有嵌套的矩形框
# 		{
# 			Rect r = found[i];
# 			int j = 0;
# 			for (; j < found.size(); j++)
# 			{
# 				if ( i != j && (r & found[j]) == r)
# 				{
# 					break;
# 				}
# 			}
# 			if(j == found.size())
# 				foundFiltered.push_back(r);
# 		}
# 		for( int i = 0; i < foundFiltered.size(); i++)//画出矩形框
# 		{
# 			Rect r = foundFiltered[i];
# 			rectangle(testImgNorm, r.tl(), r.br(), Scalar(0,255,0), 1);
# 		}
#
# 		imshow("test",testImgNorm);
# 		waitKey();
# 	}
#
# 	system("pause");
#
# }
"""

import cv2
import numpy as np


def getHOGDetector(svm_model):
    """获取检测子，即cv::HOGDescriptor::setSVMDetector()的参数"""
    support_vector = svm_model.getSupportVectors()
    rho, _, _ = svm_model.getDecisionFunction(0)
    support_vector = np.transpose(support_vector)

    return np.append(support_vector, [[-rho]], 0)


if __name__ == "__main__":
    # 使用python实现上面C++代码
    svm_model_path = 'SVM_HOG.xml'
    # python 好像支持的是.mat的数据文件的保存，好像不支持.xml文件（C++保存的格式）
    svm = cv2.ml.SVM_load(svm_model_path)

    pass
