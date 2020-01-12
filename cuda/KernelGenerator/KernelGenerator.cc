# include "KernelGenerator.hh"

KernelGenerator::KernelGenerator(KernelType k, size_t rows, size_t cols)
{
    this->type_ = k;
    this->rows_ = rows;
    this->cols_ = cols;
}

int** KernelGenerator::generate()
{
    switch (this->type_)
    {
    case KernelType::Circle :
        return this->generateCircle();
    case KernelType::Square :
        return this->generateSquare();
    default:
        return nullptr;
    }
}

int** KernelGenerator::generateSquare()
{
    int** kernel = new int*[this->rows_];

    for (int i = 0; i < this->rows_; i++) {
        kernel[i] = new int[this->cols_] {};
        for (int j = 0; j < this->cols_; j++) {
            kernel[i][j] = 1;
        }
    }
    return kernel;

}

int** KernelGenerator::generateCircle()
{
    int** kernel = new int*[this->rows_];
    int radius = this->rows_;
    radius /= 2;
    for (int i = 0; i < this->rows_; i++) {
        kernel[i] = new int[this->cols_] {};
        for (int j = 0; j < this->cols_; j++) 
             kernel[i][j] = (sqrt(pow(i - radius, 2) + pow(j - radius, 2)) <= radius)? 1.0 : 0.0;
    }

    return kernel;
}

void KernelGenerator::type_set(KernelType k) 
{
    this->type_ = k;
}

void KernelGenerator::rows_set(size_t rows) 
{
    this->rows_ = rows;
}

void KernelGenerator::cols_set(size_t cols) 
{
    this->cols_ = cols;
}


void displayMatrix(int** res, size_t rows, size_t cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << res[i][j] << " ";
		}
		std::cout << std::endl;
	}
    
}



/*
        KernelType type_;
        size_t rows_;
        size_t cols_
*/