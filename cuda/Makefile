CC=nvcc 
NVCCFLAGS = -ccbin g++-7  
CPPFLAGS= -Wall -Wextra -Werror -std=c++17 -pedantic -g 
OPENCV = $(shell pkg-config --cflags --libs opencv)

VPATH = ./ImageHandler/ \
       	./KernelGenerator/	

OBJ = ImageHandler.o KernelGenerator.o

BIN = main

INC = -I/home/dude/file/dude/project/gpgpu/GPGPU_Project/cuda/ImageHandler \
      -I/home/dude/file/dude/project/gpgpu/GPGPU_Project/cuda/KernelGenerator


${BIN} : OBJ 
	${CC} ${NVCCFLAGS} ${INC} ${OBJ} ${BIN}.cu -o ${BIN} ${OPENCV}
OBJ : 
	g++ ${CPPFLAGS} ${INC} -c ImageHandler/ImageHandler.cc ${OPENCV} 
	g++ ${CPPFLAGS} ${INC} -c KernelGenerator/KernelGenerator.cc 
clean  :
	${RM} -rf ${BIN} ImageHandler.o KernelGenerator.o
	
