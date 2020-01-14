#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;

enum class Kernel {
	Rectangle,
	Circle
};

void displayMatrix(int** res, int h, int w) {
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << res[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

// generate kernel according to kernel type

int** get_kernel(const Kernel type, int height, int width) {
	
	int** kernel = new int*[height];
	
	for (int i = 0; i < height; i++) {
		kernel[i] = new int[width] {};
		for (int j = 0; j < width; j++) {
			kernel[i][j] = 1;
		}
	}

	if (type == Kernel::Rectangle) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				kernel[i][j] = 1;
			}
		}
	}

	if (type == Kernel::Circle) {
		//FIXME
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				kernel[i][j] = 1;
			}
		}
	}
	
	return kernel;
}

// Erosion

int kernelMatchErosion(int** src, int** kernel, int rows, int cols, int x, int y, int w, int z) {
	int match = 1;
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			if ((w+i) >= rows || (z+j) >= cols) {
				match = 0;
			}
			else {
				if (src[w + i][z + j] != kernel[i][j]) {
					match = 0;
				}
			}
		}
	}
	return match;
}

int** erosion(int** src, Kernel type, int rows, int cols, int height, int width, int stride = 1) {
	int** kernel = get_kernel(type, height, width);
	displayMatrix(kernel, height, width);
	std::cout << std::endl;
	
	int** res = new int*[rows];
	
	for (int i = 0; i < rows; i++) {
		res[i] = new int[cols] {};
	}

	for (int i = 0; i < rows; i+=stride) {
		for (int j = 0; j < cols; j+=stride) {
			if (kernelMatchErosion(src, kernel, rows, cols, height, width, i, j)) {
				res[i + (height / 2)][j + (width / 2)] = 1;
			}
		}
	}
	return res;
}


// dilatation

int kernelMatchDilatation(int** src, int** kernel, int rows, int cols, int x, int y, int w, int z) {
	int match = 0;
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			if ((w+i) < rows && (z+j) < cols) {
				if (kernel[i][j] == src[w + i][z + j]) {
					match = 1;
				}
			}
		}
	}
	return match;
}

int** dilatation(int** src, Kernel type, int rows, int cols, int height, int width, int stride = 1) {
	int** kernel = get_kernel(type, height, width);
	displayMatrix(kernel, height, width);
	std::cout << std::endl;
	
	int** res = new int*[rows];
	
	for (int i = 0; i < rows; i++) {
		res[i] = new int[cols] {};
	}

	for (int i = 0; i < rows; i+=stride) {
		for (int j = 0; j < cols; j+=stride) {
			if (kernelMatchDilatation(src, kernel, rows, cols, height, width, i, j)) {
				if ((i + (height / 2)) < rows && (j + (width / 2)) < cols) {
					res[i + (height / 2)][j + (width / 2)] = 1;
				}
			}
		}
	}
	return res;
}

void mat2Image(int** mat, int rows, int cols, cv::Mat* res)
{
    //std::cout << "rows : " << rows << " cols : " << cols << std::endl;
    for (int i = 0; i < rows; i++)
	{
        for (int j = 0; j < cols; j++)
        {   
            if (mat[i][j] == 1)
                res->at<uchar>(i, j) = 0.;
        }
	}
}


int main(int argc, char** argv )
{

	auto start = std::chrono::high_resolution_clock::now();
	
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );
	cv::Mat greyMat, binaryMat;
	cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);
	cv::threshold(greyMat, binaryMat, 150, 255, cv::THRESH_BINARY);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", binaryMat);

    //waitKey(0);

	int** binaryImage = new int*[greyMat.rows];
	std::cout << "Start binarization..." << std::endl;
	for (int i = 0; i < greyMat.rows; i++) {
		binaryImage[i] = new int[greyMat.cols] {0};
		for (int j = 0; j < greyMat.cols; j++) {
			int r = binaryMat.ptr(i, j)[0];
			int g = binaryMat.ptr(i, j)[1];
			int b = binaryMat.ptr(i, j)[2];
			if (!(r == 255 && g == 255 && b == 255)) {
				binaryImage[i][j] = 1;
			}
		}
	}
	//displayMatrix(binaryImage, binaryMat.rows, binaryMat.cols);

	std::cout << std::endl;

	Kernel type = Kernel::Rectangle;
	int**  res = dilatation(binaryImage, type, binaryMat.rows, binaryMat.cols, 5, 5);
	//displayMatrix(res, binaryMat.rows, binaryMat.cols);

	//int** mat, int rows, int cols, cv::Mat* res

	cv::Mat resMat = cv::Mat(binaryMat.rows, binaryMat.cols,CV_8UC1, cv::Scalar(255));
	mat2Image(res, binaryMat.rows, binaryMat.cols, &resMat);
	//cv::imshow("Result", resMat);

    //cv::waitKey(0);
    cv::imwrite("some.jpg", resMat);

	auto elapsed = std::chrono::high_resolution_clock::now() - start;

	long long microseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

	std::cout << "Time: " << microseconds << "ms" << std::endl; 
    return 0;
}