// Create other necessary functions her
#include<math.h>
#define block_size 16
// Matrix multiplication kernel called by MatMul()
 __global__ void GPUDriverFunction(int *A, int *B, int *o,int N)
{
 int index_row = blockIdx.x * blockDim.x + threadIdx.x;
 int i,j=0,row,col,index=index_row;
 if(index<2*N-1)
{
 o[index]=0;
 if(index>=N)
 {
 j=index%N+1;
 }
 for(i=j;i<=index-j;i++)
{
 row=i;
 col=index-i;
 o[index] += A[row * N + col] * B[col * N + N - row-1];
}
}
}
void gpuThread(int N, int *matA, int *matB, int *output)
{

    int *d_A,*d_B,*d_O;
    //cout<<"ks"<<matA[0]<<matB[0]<<endl;
    long long size = N*N * sizeof(int);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    long long size1=(2*N-1)*sizeof(int);
    cudaMalloc((void**)&d_O, size1);
      cudaMemcpyAsync(d_A, matA, size, cudaMemcpyHostToDevice);
      cudaMemcpyAsync(d_B, matB, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(block_size, 1);
    //int s=sqrt(2*N);
    dim3 dimGrid((2*N) / dimBlock.x,1);
    GPUDriverFunction<<<dimGrid, dimBlock>>>(d_A, d_B,d_O,N);
    cudaMemcpyAsync(output, d_O, size1,cudaMemcpyDeviceToHost);
    //cout<<matA[0];
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_O);
}
