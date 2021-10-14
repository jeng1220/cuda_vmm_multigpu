# CUDA Virtual Memory Management with MultiGPU #
Two examples demonstrate the interoperability between CUDA virtual memory management and multi-GPU computing

## Requirement ##
* CUDA 11
* NCCL
* MPI (NVIDIA HPC SDK is recommended)
* Two GPUs

## Build ##
```sh
$ make -j`nproc`
```

## Run ##
```sh
$ ./single_process_multigpu
$ mpirun -n 2 ./multiprocess_multigpu
```