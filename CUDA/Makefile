EXE      = correlation_cuda
SRC      = correlation.cu
HEADERS  = correlation.h

CUDAFLAGS   = -O2 -I. $(EXT_CUDAFLAGS)
CUDALDFLAGS = -lm

.PHONY: all exe clean veryclean run

all: exe

exe: $(EXE)

$(EXE): $(SRC) $(HEADERS)
	nvcc $(CUDAFLAGS) $< -o $@ $(CUDALDFLAGS)

clean:
	-rm -vf $(EXE) *~

veryclean: clean

run: $(EXE)
	./$(EXE)
