CC=g++

CPPFLAGS = -Wall -Wextra -Werror -std=c++17 -pedantic -g 

OPENCV=  $(shell pkg-config --cflags --libs opencv) 
VPATH = ../ImageHandler/
INC = -I/home/dude/file/dude/project/gpgpu/GPGPU_Project/cuda/ImageHandler 


OBJ = ImageHandler
BIN=main

${BIN} : ${OBJ} 
	${CC} ${CFLAGS} ${INC} ${BIN}.cc -o ${BIN} ${OBJ}.o ${OPENCV}
${OBJ} : 
	${CC} ${CFLAGS} ${INC} -c ${VPATH}${OBJ}.cc  ${OPENCV}

clean  :
	${RM} -rf ${BIN} ${OBJ}.o 
	
