#include <stdio.h>
# include "KernelGenerator.hh"


int main(int argc, char** argv )
{
	int** kernel;

	KernelGenerator K = KernelGenerator(KernelType::Square, 3, 3);
	std::cout << "Generating kernel of size 3x3" << std::endl;
	
	std::cout << " + Square" << std::endl;
	kernel = K.generate();
	displayMatrix(kernel, 3, 3);

	std::cout << " + Circle" << std::endl;
	K.type_set(KernelType::Circle);
	kernel = K.generate();
	displayMatrix(kernel, 3, 3);

	std::cout << "Setting size to 19x19" << std::endl;
	K.rows_set(19);
	K.cols_set(19);
	K.type_set(KernelType::Square);

	std::cout << " + Square" << std::endl;
	kernel = K.generate();
	displayMatrix(kernel, 19, 19);
	
	std::cout << " + Circle" << std::endl;
	K.type_set(KernelType::Circle);

	kernel = K.generate();
	displayMatrix(kernel, 19, 19);

	std::cout << "Setting size to 25x25" << std::endl;
	K.rows_set(25);
	K.cols_set(25);
	K.type_set(KernelType::Square);

	std::cout << " + Square" << std::endl;
	kernel = K.generate();
	displayMatrix(kernel, 25, 25);
	
	std::cout << " + Circle" << std::endl;
	K.type_set(KernelType::Circle);

	kernel = K.generate();
	displayMatrix(kernel, 25, 25);



}
