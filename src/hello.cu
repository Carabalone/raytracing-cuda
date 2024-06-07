#include <stdio.h>

__global__
void sayHello() {
  printf("Hello World %d\n", threadIdx.x);
}

// int main() {
//
//   sayHello<<<1,256>>>();
//
//   cudaDeviceSynchronize();
//
//   return 0;
// }
