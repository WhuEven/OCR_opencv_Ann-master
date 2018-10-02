#include<opencv.hpp>
#include"train.h"
#include"getData.h"
#include"getFeature.h"

using namespace cv;

int train(string data_path)
{
	Mat train_data_mat;		//ͼ������������
	Mat labels_mat;			//��Ӧ��ͼ���������

	train_data_mat.convertTo(train_data_mat, CV_32F);
	labels_mat.convertTo(labels_mat, CV_32FC1);

	/*BP ģ�ʹ����Ͳ�������*/
	Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::create();

	Mat layers_size = (Mat_<int>(1, 3) << 144, 72, 36); // 2ά�㣬1ά���
	bp->setLayerSizes(layers_size);

	bp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.1, 0.1);
	bp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
	bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 0.01));

	GetData(data_path, train_data_mat, labels_mat);		//��ȡͼ����������������Ӧ��ͼ���������


	/* ����ѵ���õ����������*/
	cout << "Start training........." << endl;
	bool trained = bp->train(train_data_mat, ml::ROW_SAMPLE, labels_mat);
	if (trained) {
		cout << "Training is over!!!!!!" << endl;
		bp->save("nn_param.xml");
		return 0;
	}
	else
	{
		cout << "Training failed!!!!!!" << endl;
		return -1;
	}

}

float test(string path)
{
	long correct_num = 0, total_num = 0;
	long cor_num_class = 0, total_num_class = 0;
	float correct_rate = 0.0f;
	ANN_Wz _ann = ANN_Wz();
	int result;

	intptr_t	hFile1 = 0;
	intptr_t	hFile2 = 0;
	struct _finddata_t fileinfo;  //����ļ���Ϣ;

	string p;
	string subDir;  //��Ŀ¼·������
	string fileName;	//�ļ�Ŀ¼����

	int classNum = 0;

	cout << "Start testing the test set...." << endl;

	if ((hFile1 = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib == _A_SUBDIR)) //�ж��Ƿ�Ϊ�ļ���
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0) //�������ѵ�����ļ���
				{		
					cor_num_class = 0;
					total_num_class = 0;
					classNum = strToInt(fileinfo.name);	

					hFile2 = 0;
					subDir.assign(path).append("\\").append(fileinfo.name);  //��Ŀ¼·��																			
					if ((hFile2 = _findfirst(p.assign(subDir).append("\\*").c_str(), &fileinfo)) != -1)		// ��ȡ��Ŀ¼�µ�һ���ļ����
					{
						do
						{
							if ((fileinfo.attrib != _A_SUBDIR))  //�ļ�Ϊͼ���ļ�
							{								
								fileName.assign(subDir).append("\\").append(fileinfo.name); //�ļ�����·��+�ļ���
								Mat temp_img = imread(fileName, 0);
								if (NULL==temp_img.data){
									continue;
								}
								result = _ann.predict(temp_img);	//����ѵ���õ�������Ԥ����

								total_num++;
								total_num_class++;
								if (result == classNum)
								{
									correct_num++;
									cor_num_class++;
								}
								else
								{
									cout << fileName << endl;
								}

							}
						} while (_findnext(hFile2, &fileinfo) == 0);  //Ѱ����һ���ļ�
					}

					correct_rate = float(cor_num_class) / float(total_num_class);//ÿһ����ȷ�ʵļ���
					cout << "Character:   " << classNum << "     " << correct_rate << endl << endl << endl;

				}
			}
		} while (_findnext(hFile1, &fileinfo) == 0);  //Ѱ����һ�����ɹ�����0������-1
	}

	correct_rate = float(correct_num) / float(total_num);//ÿһ����ȷ�ʵļ���
	cout << "Correct Rate:   " << correct_rate << endl;

	return correct_rate;
}

/*����һ����ͼ���з�ÿһ����ĸ������OCR����������*/

void img_clip(Mat comp_img) {
	ANN_Wz ann = ANN_Wz(); //����ANN������
	//Mat mul_img; //���ڴ洢�и��С��ͼ�񣬵���ʹ��
	int cols, rows;		//ԭͼ������������
	cols = comp_img.cols;
	rows = comp_img.rows;
	bool ROI_IN = false;	//�ж��Ƿ񾭹������ֲ���
	float _value = 0.0f;	//�洢��ʱ��ȡ��hist������
	int x = 0;		//���ڴ洢�иʼ�������

	int result[100];	//�洢���
	int result_num = 0;		//�ܹ��ж�����ĸ��ʶ�����


	Mat hist(1, cols, CV_32F);//�洢ͼ���������ص�sum
	float sum;//�洢ÿ���������֮��
	Mat x_img;//�������жϵ���ʱͼ�����洫��Ԥ�⺯���Ļ���ԭͼ��һ����

	cout << "clip is running" << endl;

	/*Ԥ����*/

	//ģ�����˲�
	blur(comp_img, x_img, Size(3, 3));

	//��ֵ�ָ�
	threshold(x_img, x_img, 120, 255, CV_THRESH_BINARY_INV);


	/*��ͼ��ÿ����ֵ��Ӵ洢��hist��*/
	for (int i = 0; i < cols; i++) {
		sum = 0;
		unsigned char *ptr = x_img.ptr(0)+i;
		for (int j = 0; j < rows; j++) {
			sum += *(ptr+cols*j);
		}
		hist.at<float>(i) = sum;
	}
	

	for (int i = 0; i < cols; i++)
	{
		_value = hist.at<float>(i);
		//printf("%f",_value);
		if (_value == 0) {
			//_left = _left > i ? _left : i;
			if (ROI_IN == true) 
			{
				cv::Rect _rect(x, 0, i - x,rows);//�洢�и����
				Mat mul_img = comp_img(_rect);

				//cv::imshow("�и��", mul_img);
				//cv::waitKey();
				//cv::destroyAllWindows();

				result[result_num] = ann.predict(mul_img);
				//int result = ann.predict(mul_img);
				//cout << result << endl;
				result_num++;
				ROI_IN = false;
				x = i;
			}
		}
		else {
			ROI_IN = true;
		}
	}
	int p;
	for (p = 0; p < result_num; p++) {
		cout << result[p]<<endl;
	}
	//cout << result_num<<endl;
	cout << "clip is over" << endl;
}


ANN_Wz::ANN_Wz()
{
	/*����ѵ���õ����������*/
	_Ann = ml::ANN_MLP::load("nn_param.xml");
	_feature_mat.convertTo(_feature_mat, CV_32F);	//���ڴ��ͼƬ��������
	_result = (Mat_<float>(1, CLASS_NUM));	//���ڴ��Ԥ����
}


int ANN_Wz::predict(cv::Mat img)
{
	float maxVal = -2;
	int result = 0;
	_feature_mat = getFeature(img);
	_Ann->predict(_feature_mat, _result);	//����������ȡ���
	for (int j = 0; j < CLASS_NUM; j++) {	//���������ԭ��
		float val = _result.at<float>(j);
		if (val > maxVal) {
			maxVal = val;
			result = j;
		}
	}
	
	return result;
}


ANN_Wz::~ANN_Wz()
{
}