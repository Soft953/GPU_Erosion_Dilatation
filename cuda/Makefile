CC=nvcc 
NVCCFLAGS = -ccbin g++-7  
CPPFLAGS= -Wall -Wextra -Werror -std=c++17 -pedantic -g 
OPENCV = $(shell pkg-config --cflags --libs opencv)
VPATH = ./ImageHandler/ \
       	./KernelGenerator/	
OBJ=ImageHandler
BIN=main

${BIN} : ${OBJ} 
	${CC} ${NVCCFLAGS} ${OBJ}.o ${BIN}.cu -o ${BIN} ${OPENCV}
${OBJ} : 
	g++ ${CPPFLAGS} -c ${OBJ}.cc ${OPENCV}

clean  :
	${RM} -rf ${BIN} ${OBJ}.o
	
