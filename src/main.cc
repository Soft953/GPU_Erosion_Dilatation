#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/core.hpp>


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
				std::cout << "j: " << j << " width / 2: " << width / 2 << " sum: " << j + (width / 2) << std::endl;
				res[i + (height / 2)][j + (width / 2)] = 1;
			}
		}
	}
	return res;
}

int kernelMatchDilatation(int** src, int** kernel, int rows, int cols, int x, int y, int w, int z) {
	int match = 0;
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			//std::cout << "Kernel match (" << i << "," << j << ")" << std::endl;
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
				//std::cout << "(" << i << "," << j << ")" << std::endl;
				res[i + (height / 2)][j + (width / 2)] = 1;
			}
		}
	}
	return res;
}

int main(int argc, char** argv )
{
	
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat image;
    image = imread( argv[1], 1 );

	/*
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);

    waitKey(0);
	*/
	//cv::Mat matrixTest( 13, 13, CV_8U );
	int** matrixTest = new int*[13];
	std::cout << "Start erosion..." << std::endl;
	for (int i = 0; i < 13; i++) {
		matrixTest[i] = new int[13] {};
		for (int j = 0; j < 13; j++) {
			if ((i >= 4 && i <= 8) && (j >= 3 && j <= 9)) {
				//matrixTest.at<int>(i, j) = 1;
				matrixTest[i][j] = 1;
			}
			else if (i == 1) {
				matrixTest[i][j] = 1;
			}
			else {
				matrixTest[i][j] = 0;
				//matrixTest.at<int>(i, j) = 0;
			}
		}
	}
	displayMatrix(matrixTest, 13, 13);
	std::cout << std::endl;
	Kernel type = Kernel::Rectangle;
	int**  res = dilatation(matrixTest, type, 13, 13, 3, 1);
	displayMatrix(res, 13, 13);
    return 0;
}