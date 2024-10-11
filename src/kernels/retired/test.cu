#include <stdio.h>

extern "C" __global__ void testKernel() {
    printf("Hello from kernel!\n");
}
