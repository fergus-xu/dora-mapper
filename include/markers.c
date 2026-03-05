#include "markers.h"

__attribute__((noinline, used))
void __kernel_region_start(void) {
  __asm__ __volatile__("" ::: "memory");
}

__attribute__((noinline, used))
void __kernel_region_end(void) {
  __asm__ __volatile__("" ::: "memory");
}