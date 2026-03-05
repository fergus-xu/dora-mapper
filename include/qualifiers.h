#pragma once

// Only works with clang
#if defined(__clang__)
  #define MAPPER_ANNOTATE(tag) __attribute__((annotate(tag)))
#else
  #define MAPPER_ANNOTATE(tag)
#endif

// CUDA-style equivalents
#define __host__    MAPPER_ANNOTATE("mapper.host")
#define __device__  MAPPER_ANNOTATE("mapper.device")
#define __global__  MAPPER_ANNOTATE("mapper.global")