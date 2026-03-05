#pragma once

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noinline, used))
void __kernel_region_start(void);

__attribute__((noinline, used))
void __kernel_region_end(void);

#define KERNEL_START() __kernel_region_start()
#define KERNEL_END()   __kernel_region_end()

#ifdef __cplusplus
}
#endif