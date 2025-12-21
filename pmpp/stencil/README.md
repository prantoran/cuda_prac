```bash
stencil.cu → stencil.o ──┐
                         ├→ stencil_benchmark
benchmark.cu → benchmark.o ─┘



stencil.cu → stencil.o ──┐
                         ├→ libheat_cuda.so
heat_interface.cu → heat_interface.o ─┘
```
