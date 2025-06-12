# HPC Project

This project is an university assignment for a High Performance Computing. The objective is to **optimize** the computation of a statistical correlation matrix using **OpenMP** and **CUDA** approaches, comparing their performances.

---

# :computer: Usage
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

**This is an example for copy/paste of the code compiled with tasking method, using 4000x4000 matrix and getting the execution time in output**

```bash
make clean && make run EXT_CFLAGS="-DPOLYBENCH_TIME -DPARALLEL_TASK -DEXTRALARGE_DATASET"
```

For target implementation
```bash
cd ~/spack
. share/spack/setup-env.sh
spack load llvm
make clean && make run PARALLEL_TARGET=1 EXT_CFLAGS="-DPOLYBENCH_TIME -DPARALLEL_TARGET -DEXTRALARGE_DATASET"
```

For CUDA this is an example with benchmark on 4000x4000 matrix
```bash
make clean && make run EXT_CUDAFLAGS="-DBENCHMARK -DEXTRALARGE_DATASET"
```

---

## Problem Description

Given a matrix of numerical data, the objective is to compute the **correlation coefficient** between each pair of columns.

The computation includes:
- Mean calculation
- Standard deviation
- Data normalization
- Correlation matrix (dot product between normalized column vectors)

---

## Technologies Used

| Language | Technology | Purpose            |
|----------|------------|--------------------|
| C        | OpenMP     | CPU parallelism    |
| CUDA     | NVIDIA GPU | GPU parallelism    |
| Polybench| Profiling  | Performance analysis|

---

## Results of speedup

| Method         | Max Speedup (vs Sequential) |
|----------------|-----------------------------|
| **Sequential** | x1 (baseline)               |
| **OpenMP**     | ~x35                        |
| **CUDA**       | ~x549                       |

---

## Implementation

### OpenMP (Correlation.c)
- Implemented three strategies:
  - `#pragma omp parallel for`
  - `#pragma omp task`
  - `#pragma omp target` (offload to GPU)
- Best performance with `target` offload
- Efficient for medium-size datasets

### CUDA (Correlation.cu)
- Used Pageable, Pinned and UVM approaches
- Matrix represented as 1D arrays for optimal memory handling
- Very good performances, especially on large datasets

---

## Profiling

Using **Polybench**, the most time-consuming part was the **correlation matrix computation**. Optimization efforts were focused here, but all stages were parallelized to ensure better scalability.

---

## Conclusions

- **CUDA offers best speedup**, but is more complex to implement.
- **OpenMP is easier and give good speedup**.