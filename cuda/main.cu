# include <opencv2/opencv.hpp>
# include <stdio.h>
# include <cuda.h>

# include "ImageHandler.hh"
# include "KernelGenerator.hh"


using namespace cv;

__global__ void applyKernel(int* data, int* kernel, size_t len_kernel, int* res)
{
    //cheking boudaries
    auto dim = gridDim.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    auto D = ([dim, data](int i, int j) {return data[i * dim + j];});


    int half_len = len_kernel / 2;

    if (i - half_len >= 0 && i + half_len <= blockDim.x && 
        j - half_len >= 0 && j + half_len <= blockDim.x)
    {

        auto K = ([len_kernel, kernel](int i, int j) {return kernel[i * len_kernel + j];});

        for (int line = -half_len; line < 2 * half_len; line++)
        {
            for (int col = -half_len; col < 2 * half_len; col++)
            {
                int value = 1;
                if (!(K(line + half_len, col + half_len) == D(i + line, j + col) == 1))
                {
                    res[i * dim  + j] = 0;
                    return;
                }
                
            }
        }
        res[i * dim  + j] = 1;


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
    
    std::cout << "Image of size " << rows << "x" << cols << std::endl;

	int* binary_vec = handler.mat2Vec(binary);
	int** binary_mat = handler.vec2Mat(binary_vec, rows, cols);

    // Display matrix
	//handler.display_matrix(binary_mat, rows, cols);

    std::cout << std::endl;
    
    // Setting up kernelparameters
    Kernel type = Kernel::Square;
    size_t k_rows = 3;
    size_t k_cols = 3;
    
    // Generate Kernel
    KernelGenerator KG = KernelGenerator(type, k_rows, k_cols);
    int** kernel = KG.generate();
    int*  kernel_vec = handler.mat2Vec(kernel, k_rows, k_cols);

    // Display Kernel
    std::cout << "Kernel of size " << k_rows << "x" << k_cols << std::endl;
    handler.display_matrix(kernel, k_rows, k_cols);

    /* Allocate Device memory */
    size_t size_kernel = k_cols * k_rows * sizeof(int);
    size_t size_data   = cols * rows * sizeof(int);

    // Data
    int* d_binary_vec;
    cudaMalloc((void **) &d_binary_vec, size_data);

    // Kernel
    int* d_kernel_vec;
    cudaMalloc((void **) &d_kernel_vec, size_kernel);

    // Return
    int* d_result;
    cudaMalloc((void **) &d_result, size_data);

    /* Copy Data */
    
    // Data
    cudaMemcpy(d_binary_vec, binary_vec, size_data, cudaMemcpyHostToDevice);
   
    // Kernel
    cudaMemcpy(d_kernel_vec, kernel_vec, size_kernel, cudaMemcpyHostToDevice);

    // Calling kernel
    std::cout << "calling kernel with " << rows << " block sand " << cols << " threads" << std::endl;
    applyKernel<<<rows, cols>>>(d_binary_vec, d_kernel_vec, k_rows, d_result);
    cudaDeviceSynchronize();
      
    // Copy back result
    std::cout << "Copy back vector from device memory. " << rows * cols << " elements." << std::endl;
    int* result = new int[rows * cols];

    cudaMemcpy(result, d_result, size_data , cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_binary_vec);
    cudaFree(d_kernel_vec);
    cudaFree(d_result);

    int** result_mat = handler.vec2Mat(result, rows, cols);

    handler.display_matrix(result_mat, rows, cols);

    cv::Mat res = cv::Mat(rows, cols,CV_8UC1, cv::Scalar(255));
    std::cout << "nb channels : " << res.channels() << std::endl;
    std::cout << "height : " << res.size().height << "width : " << res.size().width << std::endl;
    handler.mat2Image(result_mat, rows, cols, &res);


    cv::imshow("Result BITCH", res);

    cv::waitKey(0);
    cv::imwrite("some.jpg", res);
    return 0;
}
