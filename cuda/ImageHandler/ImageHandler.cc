# include "ImageHandler.hh"

ImageHandler::ImageHandler(char* path, char* name)
{
    this->path_ = path;
    this->name_ = name;
    // Loading Image
    this->image_ = cv::imread(this->path_, 1);
    
    if (!this->image_.data)
    {
        std::cerr << "Image at" << std::string(this->path_) << " contains no data. Stopping now." << std::endl;
        exit(-1);
    }

    // convert to greysclare
    cv::cvtColor(this->image_, this->greyscale_mat_, cv::COLOR_BGR2GRAY);
    
    // convert to binary
	cv::threshold(this->greyscale_mat_, this->binary_mat_, 150, 255, cv::THRESH_BINARY);


}

void ImageHandler::display_matrix(int** mat, int h, int w)
{
    for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			std::cout << mat[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

void ImageHandler::display_image()
{
    cv::imshow(std::string(this->path_), this->binary_mat_);
    cv::imwrite(std::string("ref") + std::string(this->name_) + std::string(".jpg"), this->binary_mat_);

    cv::waitKey(0);

}

int* ImageHandler::mat2Vec(cv::Mat mat)
{
    int rows = mat.rows;
    int cols = mat.cols;

    int* res = new int[rows * cols];

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            int r = mat.ptr(i, j)[0];
            int tmp = 0;

            if (!(r == 255 ))
			{
            	tmp = 1;
			}

            res[i * cols + j] =  tmp;
        }

    return res;
}

int* ImageHandler::mat2Vec(int** mat, int rows, int cols)
{
    int* res = new int[rows * cols];

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            res[i * cols + j] =  mat[i][j];
        

    return res;
}

int** ImageHandler::vec2Mat(int* vector, int h, int w)
{
    int** res = new int*[h];
    for(int i = 0; i < h; ++i)
        res[i] = new int[w];

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            res[i][j] = vector[i * w + j];

    return res;
}

void ImageHandler::mat2Image(int** mat, int rows, int cols, cv::Mat* res)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {   
            if (mat[i][j] == 1)
                res->at<uchar>(i, j) = 0.;
            

        }
}



cv::Mat ImageHandler::image_get()
{
    return this->image_;
}

cv::Mat ImageHandler::greyscale_get()
{
    return this->greyscale_mat_;
}

cv::Mat ImageHandler::binary_get()
{
    return this->binary_mat_;
}

/*
        cv::Mat image_;
        char* path_;
        cv::Mat greyscale_mat_;
        cv::Mat binary_mat_;

*/