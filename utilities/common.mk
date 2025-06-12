INCPATHS = -I$(UTIL_DIR)

EXE = correlation_acc
SRC = correlation.c
HEADERS = correlation.h

SRC += $(UTIL_DIR)/polybench.c

DEPS        := Makefile.dep
DEP_FLAG    := -MM

LD=ld
OBJDUMP=objdump

ifdef PARALLEL_TARGET
CC = clang
OPT=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O3
CFLAGS = $(OPT) -I. $(EXT_CFLAGS)
else
CC=gcc
OPT=-O2 -g -fopenmp
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
endif
LDFLAGS=-lm $(EXT_LDFLAGS)

.PHONY: all exe clean veryclean

all : exe

exe : $(EXE)

$(EXE) : $(SRC)
	$(CC) $(CFLAGS) $(INCPATHS) $^ -o $@ $(LDFLAGS)

clean :
	-rm -vf -vf $(EXE) *~ 

veryclean : clean
	-rm -vf $(DEPS)

run: $(EXE)
	./$(EXE)

$(DEPS): $(SRC) $(HEADERS)
	$(CC) $(INCPATHS) $(DEP_FLAG) $(SRC) > $(DEPS)

-include $(DEPS)