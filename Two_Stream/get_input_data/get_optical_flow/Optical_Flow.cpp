#include "opencv2/video/tracking.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <string>
#include <Window.h>
#include <Shlwapi.h>
#include <io.h>
#include <fstream>
#include <sstream>
#include "FindFiles.h"
using namespace cv;
using namespace std;

vector<string> get_folder_file_Name(string route) //name[0]->folder name[1]->file
{
	vector<string> name;
	int pos1 = route.find('/');
	int pos2 = route.rfind('/');

	name.push_back(route.substr(pos1 + 1, pos2 - pos1 - 1));
	name.push_back(route.substr(pos2 + 1, route.size() - 1 - pos2));

	return name;
}

string convert_int_to_string(int x)
{
	stringstream ss;
	string st;

	ss << x;
	ss >> st;

	return st;
}

void colorChange(Mat& image)
{
	int nr = image.rows;
	int nc = image.cols;

	for (int j = 0; j<nr; j++) {
		float* data = image.ptr<float>(j);
		for (int i = 0; i<nc; i++) {
			data[i] *= 255;
		}
	}
}

void createDir(string dirpath)
{
	CreateDirectory(dirpath.c_str(), NULL);
}

int main()
{
	int frame_counts;
	char path[1000] = ".\\Videos";

	FindFiles ff;
	vector<string> fileNames;

	fileNames = ff.findFiles(path);
	
	for (const auto& filename : fileNames)
	{
		VideoCapture capture(filename);
		Mat prevgray, gray, flow, frame;
		vector<string> folder_file_name = get_folder_file_Name(filename);

		frame_counts = 0;

		cout << folder_file_name[1] << endl;
		do
		{
			capture >> frame;
			try
			{
				cvtColor(frame, gray, CV_RGB2GRAY);
			}
			catch (...)
			{
				break;
			}

			if (prevgray.data)
			{
				calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 1, 15, 3, 5, 1.2, 0);

				vector<Mat> v;
				split(flow, v);

				colorChange(v[0]);
				colorChange(v[1]);

				string horizontal_name = ".\/Result\\" + folder_file_name[0] + "\\" + folder_file_name[1] + "\\" + convert_int_to_string(frame_counts) + "_horizontal.jpg";
				string vertical_name = ".\/Result\\" + folder_file_name[0] + "\\" + folder_file_name[1] + "\\" + convert_int_to_string(frame_counts) + "_vertical.jpg";
				string dirpath1 = ".\/Result\\" + folder_file_name[0];
				string dirpath2 = ".\/Result\\" + folder_file_name[0] + "\\" + folder_file_name[1];
				
				createDir(dirpath1);
				createDir(dirpath2);

				imwrite(horizontal_name, v[0]);
				imwrite(vertical_name, v[1]);
			}
			if (waitKey(100) >= 0)
			{
				break;
			}
			swap(prevgray, gray);

			frame_counts++;

		} while (true);
	}
	system("pause");

	return 0;
}