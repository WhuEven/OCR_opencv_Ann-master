#include"train.h"
#include<opencv.hpp>

using namespace std;
using namespace cv;


/*���ֻ�ΪӢ����ĸ���*/
void num_to_char(int num)
{
	if (num < 10)
		cout << num << endl;
	else
	{
		num -= 10;
		num += 'A';
		cout << (char)num << endl;
	}
}


int main()
{

	//getFeature("E:\\OCR\\Train\\B\\10-5.jpg");	//��ȡͼ����������

		
	/*Mat train_data_mat;		//ͼ������������
	Mat labels_mat;			//��Ӧ��ͼ���������
	train_data_mat.convertTo(train_data_mat, CV_32F);
	GetData("E:\\OCR\\Train", train_data_mat, labels_mat);*/


	//train("E:\\OCR\\Train1"); //ѵ��


	/*
	//����ͼƬ����
	char filename[100];
	ANN_Wz ann = ANN_Wz();
	int result;
	while (1) {
		gets_s(filename);
		Mat img = imread(filename, 0);
		if (NULL==img.data)
		{
			cout << "NULL" << endl;
			continue;
		}
		result = ann.predict(img);
		num_to_char(result);
		//cout << result << endl;
	}
	*/


	//float rate;
	//rate=test("E:\\OCR\\Train");

	
	//һ���ַ�����
	Mat img = cv::imread("E:\\OCR\\TEST1.png", 0);
	//cv::imshow("image",img);
	//cv::waitKey();
	//cv::destroyAllWindows();
	img_clip(img);

	
	system("pause");
	

	return 0;
   }
