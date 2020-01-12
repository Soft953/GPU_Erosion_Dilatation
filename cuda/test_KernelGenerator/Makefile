CC=g++

CPPFLAGS = -Wall -Wextra -Werror -std=c++17 -pedantic -g 

VPATH = ../KernelGenerator/
INC = -I/home/dude/file/dude/project/gpgpu/GPGPU_Project/cuda/KernelGenerator 


OBJ = KernelGenerator
BIN=main

${BIN} : ${OBJ} 
	${CC} ${CFLAGS} ${INC} ${BIN}.cc -o ${BIN} ${OBJ}.o 
${OBJ} : 
	${CC} ${CFLAGS} ${INC} -c ${VPATH}${OBJ}.cc  

clean  :
	${RM} -rf ${BIN} ${OBJ}.o
	
