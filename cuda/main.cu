# include <opencv2/opencv.hpp>
# include <stdio.h>
# include <cuda.h>
# include <chrono> 

# include "ImageHandler.hh"
# include "KernelGenerator.hh"


#define ErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace cv;

__global__ void gpu_erosion(int* data, int* kernel, int k_rows, int k_cols, int rows, int cols, int* res)
{
    
    int xgridOffet = rows / gridDim.x;
    int ygridOffet = cols / gridDim.y;

    int xGrid = blockIdx.x;
    int yGrid = blockIdx.y;

    int xthread = threadIdx.x;
    int ythread = threadIdx.y;

    int i = xthread + xGrid * xgridOffet;
    int j = ythread + yGrid * ygridOffet;

    if (i - k_rows/2 >= 0 && i + k_rows/2 < rows && 
        j - k_cols/2 >= 0 && j + k_cols/2 < cols)
    {
        //printf("dsffdsdsf");
        auto D = ([cols, data](int i, int j) {return data[i * cols + j];});
        auto K = ([k_cols, kernel](int i, int j) {return kernel[i * k_cols + j];});

        for (int line = -k_rows/2; line <= k_rows/2; line++)
        {
            for (int col = -k_cols/2; col <= k_cols/2; col++)
            {
                if (!(K(line + k_rows/2, col + k_cols/2) == D(i + line, j + col) == 1))
                {
                    res[i * cols  + j] = 0;
                    return;
                }
                
            }
        }
        res[i * cols  + j] = 1;
    }
}


__global__ void gpu_dilatation(int* data, int* kernel, int k_rows, int k_cols, int rows, int cols, int* res)
{
    
    int xgridOffet = rows / gridDim.x;
    int ygridOffet = cols / gridDim.y;

    int xGrid = blockIdx.x;
    int yGrid = blockIdx.y;

    int xthread = threadIdx.x;
    int ythread = threadIdx.y;

    int i = xthread + xGrid * xgridOffet;
    int j = ythread + yGrid * ygridOffet;

    if (i >= 0 && i < rows && j >= 0 && j < cols)
    {
        auto D = ([cols, rows, data](int i, int j) {
            if (i >= 0 && i < rows && j >= 0 && j < cols)
                return data[i * cols + j];
            else
                return 0;});
        auto K = ([k_cols, kernel](int i, int j) {return kernel[i * k_cols + j];});

        for (int line = -k_rows/2; line <= k_rows/2; line++) 
        {
            for (int col = -k_cols/2; col <= k_cols/2; col++) 
            {
                if (K(line + k_rows/2, col + k_cols/2) && D(i + line, j + col) == 1)
                {
                    res[i * cols  + j] = 1;
                    return;
                }
                
            }
        }
        res[i * cols  + j] = 0;
    }
}


int* erosionNaive(int k_rows, int k_cols, int rows, int cols, KernelType type, int* binary_vec, ImageHandler handler)
{
        
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
    std::cout << "Allocating data to Device" << std::endl;

    // Data
    std::cout << "\to Data " << size_kernel << std::endl;
    int* d_binary_vec;
    cudaMalloc((void **) &d_binary_vec, size_data);

    // Kernel
    std::cout << "size kernel : " << size_kernel << std::endl;
    int* d_kernel_vec;
    cudaMalloc((void **) &d_kernel_vec, size_kernel);

    // Return
    int* d_result;
    cudaMalloc((void **) &d_result, size_data);

    /* Copy Data */
    std::cout << "Copying Data to Device" << std::endl;
    // dst src
    // Data
    cudaMemcpy(d_binary_vec, binary_vec, size_data, cudaMemcpyHostToDevice);

    // Kernel
    cudaMemcpy(d_kernel_vec, kernel_vec, size_kernel, cudaMemcpyHostToDevice);

    // Processing Kernel Dimension

    dim3 DimBlock(32, 32);
    std::cout << "dimBlock : " << DimBlock.x << "x" << DimBlock.y << std::endl;


    int xGrid = (int) (cols / DimBlock.x);
    int yGrid = (int) (rows / DimBlock.y);

    std::cout << "dimGrid : " << xGrid << "x" << yGrid << std::endl;
    dim3 DimGrid(yGrid, xGrid);


    // Calling kernel
    std::cout << "calling kernel with " << xGrid << " blocks and " << yGrid << " threads" << std::endl;
    auto start = std::chrono::steady_clock::now();

    gpu_erosion<<<DimGrid, DimBlock>>>(d_binary_vec, d_kernel_vec, k_rows, k_cols, rows, cols, d_result);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();


    std::cout << "Elapsed time in milliseconds : " 
    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    << " ms" << std::endl;


    ErrorCheck(cudaPeekAtLastError());
    
    // Copy back result
    std::cout << "Copy back vector from device memory. " << rows * cols << " elements." << std::endl;
    int* result = new int[rows * cols];

    cudaMemcpy(result, d_result, size_data , cudaMemcpyDeviceToHost);

    return result;
}

int* dilatationNaive(int k_rows, int k_cols, int rows, int cols, KernelType type, int* binary_vec, ImageHandler handler)
{
        
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
    std::cout << "Allocating data to Device" << std::endl;

    // Data
    std::cout << "\to Data " << size_kernel << std::endl;
    int* d_binary_vec;
    cudaMalloc((void **) &d_binary_vec, size_data);

    // Kernel
    std::cout << "size kernel : " << size_kernel << std::endl;
    int* d_kernel_vec;
    cudaMalloc((void **) &d_kernel_vec, size_kernel);

    // Return
    int* d_result;
    cudaMalloc((void **) &d_result, size_data);

    /* Copy Data */
    std::cout << "Copying Data to Device" << std::endl;
    // dst src
    // Data
    cudaMemcpy(d_binary_vec, binary_vec, size_data, cudaMemcpyHostToDevice);

    // Kernel
    cudaMemcpy(d_kernel_vec, kernel_vec, size_kernel, cudaMemcpyHostToDevice);

    // Processing Kernel Dimension

    dim3 DimBlock(32, 32);
    std::cout << "dimBlock : " << DimBlock.x << "x" << DimBlock.y << std::endl;


    int xGrid = (int) (cols / DimBlock.x);
    int yGrid = (int) (rows / DimBlock.y);

    std::cout << "dimGrid : " << xGrid << "x" << yGrid << std::endl;
    dim3 DimGrid(yGrid, xGrid);

    // Calling kernel
    std::cout << "calling kernel with " << xGrid << " blocks and " << yGrid << " threads" << std::endl;
    gpu_dilatation<<<DimGrid, DimBlock>>>(d_binary_vec, d_kernel_vec, k_rows, k_cols, rows, cols, d_result);
    cudaDeviceSynchronize();


    ErrorCheck(cudaPeekAtLastError());
    
    // Copy back result
    std::cout << "Copy back vector from device memory. " << rows * cols << " elements." << std::endl;
    int* result = new int[rows * cols];

    cudaMemcpy(result, d_result, size_data , cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_binary_vec);
    cudaFree(d_kernel_vec);
    cudaFree(d_result);

    return result;
}

int* erosionOptimized(int k_rows, int k_cols, int rows, int cols, KernelType type, int* binary_vec, ImageHandler handler)
{

    // Processing Kernel call Dimension

    dim3 DimBlock(32, 32);
    std::cout << "dimBlock : " << DimBlock.x << "x" << DimBlock.y << std::endl;


    int xGrid = (int) (cols / DimBlock.x);
    int yGrid = (int) (rows / DimBlock.y);

    std::cout << "dimGrid : " << xGrid << "x" << yGrid << std::endl;
    dim3 DimGrid(yGrid, xGrid);
        

    /* Allocate Device memory */
    size_t size_data   = cols * rows * sizeof(int);
    std::cout << "Allocating data to Device" << std::endl;

    // Data
    std::cout << "\to Data " << size_data << std::endl;
    int* d_binary_vec;
    cudaMalloc((void **) &d_binary_vec, size_data);

    // Return
    int* d_result;
    cudaMalloc((void **) &d_result, size_data);

    /* Copy Data */
    std::cout << "Copying Data to Device" << std::endl;
    // dst src
    // Data
    cudaMemcpy(d_binary_vec, binary_vec, size_data, cudaMemcpyHostToDevice);

    // =========================================================PASS 1============================================================================

    // Generate Kernel
    KernelGenerator KG = KernelGenerator(type, k_rows, 1);
    int** kernel = KG.generate();
    int*  kernel_vec = handler.mat2Vec(kernel, k_rows, 1);

    // Allocate Kernel
    std::cout << "size kernel : " << k_rows << std::endl;
    int* d_kernel_vec;
    cudaMalloc((void **) &d_kernel_vec, k_rows * sizeof(int));

    // Copy Kernel
    cudaMemcpy(d_kernel_vec, kernel_vec, k_rows * sizeof(int), cudaMemcpyHostToDevice);



    // Calling kernel
    std::cout << "calling kernel with " << xGrid << " blocks and " << yGrid << " threads" << std::endl;
    auto start1 = std::chrono::steady_clock::now();
    gpu_erosion<<<DimGrid, DimBlock>>>(d_binary_vec, d_kernel_vec, k_rows, 1, rows, cols, d_result);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::steady_clock::now();


    // =========================================================PASS 2============================================================================


    // Generate Kernel
    KG.rows_set(1);
    KG.cols_set(k_cols);

    kernel = KG.generate();
    kernel_vec = handler.mat2Vec(kernel, 1, k_cols);

    // Allocate Kernel
    std::cout << "size kernel : " << k_cols << std::endl;
    cudaMalloc((void **) &d_kernel_vec, k_cols * sizeof(int));

    // Copy Kernel
    cudaMemcpy(d_kernel_vec, kernel_vec, k_cols * sizeof(int), cudaMemcpyHostToDevice);

    // Calling kernel
    std::cout << "calling kernel with " << xGrid << " blocks and " << yGrid << " threads" << std::endl;
    auto start2 = std::chrono::steady_clock::now();
    gpu_erosion<<<DimGrid, DimBlock>>>(d_result, d_kernel_vec, 1, k_cols, rows, cols, d_result);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::steady_clock::now();


    std::cout << "Elapsed time in milliseconds : " 
    << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() + std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() 
    << " ms" << std::endl;



    ErrorCheck(cudaPeekAtLastError());
    
    // Copy back result
    std::cout << "Copy back vector from device memory. " << rows * cols << " elements." << std::endl;
    int* result = new int[rows * cols];

    cudaMemcpy(result, d_result, size_data , cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_binary_vec);
    cudaFree(d_kernel_vec);
    cudaFree(d_result);

    return result;
}

int main(int argc, char** argv )
{
	printf("%d", argc);
    if ( argc != 3 )
    {
        printf("usage: ./main Image_Path name\n");
        return -1;
    }

    std::cout << "System summary" << std::endl;
    int devId = 0;
    // There may be more devices!
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devId);
    printf("Maximum grid dimensions: %d x %d x %d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum block dimensions: %d x %d x %d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);

    // Loading Image
	ImageHandler handler = ImageHandler(argv[1], argv[2]);

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
    KernelType type = KernelType::Square;
    int k_rows = 5;
    int k_cols = 5;
    

    int* result = dilatationNaive(k_rows, k_cols, rows, cols, type, binary_vec, handler);


    int** result_mat = handler.vec2Mat(result, rows, cols);

    //handler.display_matrix(result_mat, rows, cols);

    cv::Mat res = cv::Mat(rows, cols,CV_8UC1, cv::Scalar(255));
    handler.mat2Image(result_mat, rows, cols, &res);


    cv::imshow("Result BITCH", res);

    cv::waitKey(0);
    cv::imwrite(std::string(argv[2]) + std::string(".jpg"), res);
    return 0;
}
