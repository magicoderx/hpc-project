# HPC Project

## :computer: Usage
To compile and run the program normally, use this:
```bash
make clean && make run
```
---
You can also specify to use polybench to get execution times

```bash
make clean && make run EXT_CFLAGS="-DPOLYBENCH_TIME"
```
or adding `-DPOLYBENCH_FUNCTION` if you want to get execution time for each "function"

---

Another flags you can set are the dataset size chosen between
- `MINI_DATASET`: 32x32
- `SMALL_DATASET`: 500x500
- `STANDARD_DATASET`: 1000x1000 (This is the default dataset)
- `LARGE_DATASET`: 2000x2000
- `EXTRALARGE_DATASET`: 4000x4000

And also the desidered method of parallelization
- Sequential is the default
- `PARALLEL_FOR`, parallelize with `#pragma omp parallel for`
- `PARALLEL_TASK`, parallelize using tasks
- `PARALLEL_TARGET`, parallelize using GPU offloading

**This is an example for copy/paste of the code compiled with offloading method, using 4000x4000 matrix and getting the execution time in output**

```bash
make clean && make run EXT_CFLAGS="-DPOLYBENCH_TIME -DPARALLEL_TARGET -DEXTRALARGE_DATASET"
```