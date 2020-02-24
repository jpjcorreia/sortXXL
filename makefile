##
# sortXXL Makefile
# 
# author Vitor Carreira
# date 2009-09-23
# 
# author Cláudio Esperança
# date 2012-10-22
##

## Loads the configuration file
include ./configs/makefile.inc

## Flags to the CC compiler
CFLAGS=${LIBS}

## Flags to the compiler
NVCCFLAGS=${LIBS}

## Cuda compiler binary
NVCC:=nvcc

## Flags to code indentation
IFLAGS=-br -brs -npsl -ce -cli4

## Directories with the project source code
INCLUDE_DIRS=${SRC_DIR_3RD} ${EXTRA_INCLUDE_DIRS} ${SRC_DIR_INCLUDE} ${SRC_DIR}

## Generates a list of objects from the .c files on the directories specified on the variable INCLUDE_DIRS
PROGRAM_OBJS:=$(patsubst %.c,%.o,$(wildcard $(patsubst %,./%/*.c,${INCLUDE_DIRS}))) $(patsubst %.cu,%.cu.o,$(wildcard $(patsubst %,./%/*.cu,${INCLUDE_DIRS}))) 

## If the variable with the options file is set, add the object file as a dependency
ifdef PROGRAM_OPT
PROGRAM_OPT_o:=${SRC_DIR_3RD}/${PROGRAM_OPT}.o
ifeq (,$(findstring $(PROGRAM_OPT_o),$(PROGRAM_OBJS)))
PROGRAM_OBJS:=$(PROGRAM_OBJS) $(PROGRAM_OPT_o)
endif
endif

## Abstract Targets
.PHONY: clean
.PHONY: cleanall
.PHONY: cleandocs
.PHONY: gengetopt
.PHONY: debug
.PHONY: static
.PHONY: static32
.PHONY: 32

.SUFFIXES: .c .cu .o

## Compile with 32 bits support
32: CFLAGS += -m32
32: NVCCFLAGS += -m32
32: ${PROGRAM}

## Compile with 32 bits support and static linking
static32: CFLAGS += -static -m32
static32: ${PROGRAM}

## Compile with static linking
static: CFLAGS += -static
static: ${PROGRAM}

## Compile with depuration
debug: CFLAGS += -D SHOW_DEBUG -Wall -W -g -Wmissing-prototypes -Wsign-compare -Wunused-parameter -Wunused-function
debug: NVCCFLAGS += -D SHOW_DEBUG --debug -g 
debug: ${PROGRAM}

## To generate the files with gengetopt 
.${SRC_DIR_3RD}/${PROGRAM_OPT}.h: ${SRC_DIR_3RD}/${PROGRAM_OPT}.h
gengetopt: ${SRC_DIR_3RD}/${PROGRAM_OPT}.h
${SRC_DIR_3RD}/${PROGRAM_OPT}.h: configs/${PROGRAM_OPT}.ggo
	gengetopt < configs/${PROGRAM_OPT}.ggo --output-dir=${SRC_DIR_3RD}/ --file-name=${PROGRAM_OPT}

## Besides the clean target, also cleans the options files and the docs folder. Use with care!
cleanall: clean cleandocs
	@for d in $(INCLUDE_DIRS); do (cd $$d; rm -fv ${PROGRAM_OPT}.h ${PROGRAM_OPT}.c ); done

## Cleaning of the directories and subdirectories
clean:
	@for d in $(INCLUDE_DIRS); do (cd $$d; echo "Cleaning the directory '$$d':"; rm -fv *.o core.* *~ ${PROGRAM} *.bak ); done

## Remove the documentação folder
cleandocs:
	@echo "Removing the documentation folder"; rm -rfv docs

## To documentation
docs: configs/Doxyfile
	doxygen configs/Doxyfile

configs/Doxyfile:
	doxygen -g configs/Doxyfile

## From Windows do Linux
indent:
	dos2unix *.c *.h
	indent ${IFLAGS} *.c *.h

## Constructs the executable
#${PROGRAM}: gengetopt ${PROGRAM_OBJS}
${PROGRAM}: ${PROGRAM_OBJS}
	@echo "Compiling '$@':"
	$(NVCC) $(NVCCFLAGS) -o $@ ${PROGRAM_OBJS}
	
## Compile .o from .c
%.o: %.c %.h
	@echo "Construction the object '$@' with ${CC}:"
	${CC} ${CFLAGS}${EXTRA_CCFLAGS} -c -o $@ $<
	
%.cu.o: %.cu
	@echo "Construction the object '$@' with ${NVCC}:"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
