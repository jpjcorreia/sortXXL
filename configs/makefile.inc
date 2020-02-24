## Executable name
PROGRAM=sortXXL

## Default make target
all: ${PROGRAM}

## Options file from the gengetopt (comment when not available)
PROGRAM_OPT=sortXXL_cmd

## Main source code folder
SRC_DIR=./src
## Libs folder
SRC_DIR_INCLUDE=${SRC_DIR}/includes
## 3rd libs folder
SRC_DIR_3RD=${SRC_DIR}/3rd

## .c files list to use as .o to the main program
EXTRA_INCLUDE_DIRS=./ ${SRC_DIR_3RD}/libwebsockets

${SRC_DIR}/main.o: ${SRC_DIR}/main.c ${SRC_DIR}/main.h ${SRC_DIR_INCLUDE}/constants.h ${SRC_DIR_3RD}/debug.h ${SRC_DIR_3RD}/${PROGRAM_OPT}.h
${SRC_DIR_3RD}/debug.o: ${SRC_DIR_3RD}/debug.c ${SRC_DIR_3RD}/debug.h

## dependencies of the gengetopt file
${SRC_DIR_3RD}/${PROGRAM_OPT}.o: ${SRC_DIR_3RD}/${PROGRAM_OPT}.c ${SRC_DIR_3RD}/${PROGRAM_OPT}.h


## Force the compilation as 32 bits
EXTRA_CCFLAGS=# -m32

## Libraries to include
LIBS=-lcublas -lcudart -lcurand -lcrypto -lz
