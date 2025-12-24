# Chapter 8

## Code

For this chapter we implement all of the stencil kernels mentioned, in particular:
- sequential stencil
- basic parallelized stencil kernel
- stencil kernel utilizing shared memory
- stencil kernel utilizing thread coarsening
- stencil kernel utilizing register tiling

All of the kernels, alongside their host function code, can be found in [stencil.cu](stencil.cu).


Command to build and run benchmark:
```bash
nvcc benchmark.cu stencil.cu -o benchmark && ./benchmark
```

you will see

```logs
...
================================================================================
Benchmarking 3D Stencil Operations - Grid Size: 128x128x128
================================================================================
Configuration:
Grid size: 128x128x128
Total elements: 2097152
Memory per array: 8.00 MB
OUT_TILE_DIM: 8, IN_TILE_DIM: 8


Results:
Implementation           | Time (ms) | Speedup vs Sequential | Speedup vs Basic
-------------------------|-----------|----------------------|------------------
Sequential              |    4.634  |                1.00x |            0.76x
Parallel Basic          |    3.535  |                1.31x |            1.00x
Shared Memory           |    3.521  |                1.32x |            1.00x
Thread Coarsening       |    3.512  |                1.32x |            1.01x
Register Tiling         |    3.519  |                1.32x |            1.00x

Correctness Verification:
Parallel Basic vs Sequential: ✓ PASS
Shared Memory vs Sequential: ✓ PASS
Thread Coarsening vs Sequential: ✓ PASS
Register Tiling vs Sequential: ✓ PASS

Overall correctness: ✓ All implementations correct
```

### Heat simulation

We also explore the potential applications of stencil for real-world problems, and we implement a CUDA-accelerated heat diffusion simulation. The simulation leverages the `stencil_3d_parallel_register_tiling` kernel and computes the changes in heat over time.

To run it, first make sure that all of the CUDA code is already compiled. 

```bash
cd code
make
```

Then you can just run the code in [heat_simulation.py](./code/heat_simulation.py)

```bash
python heat_simulation.py
```

*Note that it might take a few minutes due to GIF being slowly created. 

The result should resemble this:

![our heat equation](./code/heat_equation_3d.gif)

