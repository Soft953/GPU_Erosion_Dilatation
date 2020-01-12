# include <opencv2/opencv.hpp>
# include <stdio.h>
# include <cuda.h>

# include "ImageHandler.hh"


using namespace cv;

__global__ void applyKernel(int* data, int* kernel, size_t len_kernel, int* res)
{
    //cheking boudaries
    int i = blockIdx.x;
    int j = threadIdx.x;

    int half_len = len_kernel / 2;

    if (i - half_len >= 0 && i + half_len <= blockDim.x && 
        j - half_len >= 0 && j + half_len <= blockDim.x)
    {

        auto D = ([data](int i, int j) {return data[i * blockDim.x + j];});
        auto K = ([len_kernel, kernel](int i, int j) {return kernel[i * len_kernel + j];});

        for (int line = -half_len; line < 2 * half_len; line++)
        {
            for (int col = -half_len; col < 2 * half_len; col++)
            {
                if (!(K(line + half_len, col + half_len) == D(i + line, j + col) == 1))
                {
                    res[D(i, j)] = 0;
                }
            }
        }
        
        res[D(i, j)] = 1;

    }
}

int main(int argc, char** argv )
{
	
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    // Loading Image
	ImageHandler handler = ImageHandler(argv[1]);

    // Display Image
	handler.display_image();

    // Convert to binary
	cv::Mat binary = handler.binary_get();

    // Convert to vector
	int rows = binary.rows;
	int cols = binary.cols;


	int* binary_vec = handler.mat2Vec(binary);
	int** binary_mat = handler.vec2Mat(binary_vec, rows, cols);

    // Display matrix
	handler.display_matrix(binary_mat, rows, cols);

    std::cout << std::endl;
    
    // Setting up kernel type
    Kernel type = Kernel::Rectangle;
    
    // Generate Kernel


    // Allocate Device memory
        // Data
    int** d_binary_vec;
    cudaMalloc((void **) &d_binary_vec, rows * cols);
        // Kernel
    int** d_kernel;


	//handler.display_matrix(res, rows, cols);
    return 0;
}