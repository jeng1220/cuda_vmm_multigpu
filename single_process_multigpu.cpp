#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#if USE_CUDA_VMM
#include "cuvector.h"
#endif

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#if USE_CUDA_VMM
static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
  if (res != CUDA_SUCCESS) {
    const char *errStr = NULL;
    (void)cuGetErrorString(res, &errStr);
    printf("Failed, CUDA Drv error %s:%d '%s'\n",
      file, line, errStr);
  }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);
#endif

int main(int argc, char* argv[])
{
  ncclComm_t comms[2];

  //managing 2 devices
  int nDev = 2;
  int size = 32*1024*1024;
  int devs[2] = { 0, 1 };

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // CUDA virtual memory management
#if USE_CUDA_VMM
  CUcontext ctx[2];
  CUdevice dev[2];
  int supportsVMM = 0;
  typedef unsigned char ElemType;
  typedef cuda_utils::Vector<ElemType, cuda_utils::VectorMemMap> VectorDUT;

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    cudaFree(0); // force runtime to create a context
    CHECK_DRV(cuCtxGetCurrent(&ctx[i]));
    CHECK_DRV(cuCtxGetDevice(&dev[i]));
    CHECK_DRV(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev[i]));
    if (not supportsVMM) {
      fprintf(stderr, "not support CUDA VMM\n");
      return 0;
    }
  }

  std::vector<VectorDUT> sendbuffDuts{
    VectorDUT(ctx[0]), VectorDUT(ctx[1])
  };

  std::vector<VectorDUT> recvbuffDuts{
    VectorDUT(ctx[0]), VectorDUT(ctx[1])
  };
#endif

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));

#if USE_CUDA_VMM
    sendbuffDuts[i].grow(size * sizeof(float));
    recvbuffDuts[i].grow(size * sizeof(float));
    sendbuff[i] = reinterpret_cast<float*>(sendbuffDuts[i].getPointer());
    recvbuff[i] = reinterpret_cast<float*>(recvbuffDuts[i].getPointer());
#else
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
#endif
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
      comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

#if !USE_CUDA_VMM
  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }
#endif

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}
