# pragma once 
# include <cstdlib>
# include <cmath>
# include <iostream>

typedef enum class Kernel {
	Square,
	Circle
}KernelType;

class KernelGenerator
{
    public : 
        KernelGenerator(KernelType k, size_t rows, size_t cols);

        int** generate();

        int** generateCircle();
        int** generateSquare();

        void type_set(KernelType k);
        void rows_set(size_t rows);
        void cols_set(size_t cols);

    private :
        KernelType type_;
        size_t rows_;
        size_t cols_;


};

void displayMatrix(int** res, size_t h, size_t w);
