# include <stdio.h>
# include <opencv2/opencv.hpp>
# include "ImageHandler.hh"

int main(int argc, char** argv )
{

	
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

	ImageHandler handler = ImageHandler(argv[1]);

	handler.display_image();

	cv::Mat binary = handler.binary_get();

	int rows = binary.rows;
	int cols = binary.cols;

	int* binary_vec = handler.mat2Vec(binary);
	int** binary_mat = handler.vec2Mat(binary_vec, rows, cols);

	handler.display_matrix(binary_mat, rows, cols);

	std::cout << std::endl;
    return 0;
}
