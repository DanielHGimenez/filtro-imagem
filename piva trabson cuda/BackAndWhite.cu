#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BlackAndWhite.h"
#include "Grayscale.h"

__global__ void black_and_white_CUDA(unsigned char* imagem, int canais);

void black_and_white(unsigned char* imagem, int altura, int largura, int canais) {
	gray_scale(imagem, altura, largura, canais);
	unsigned char* dev_imagem = NULL;

	cudaMalloc((void**)&dev_imagem, altura * largura * canais);

	cudaMemcpy(dev_imagem, imagem, altura * largura * canais, cudaMemcpyHostToDevice);

	dim3 Grid_Image(largura, altura);
	black_and_white_CUDA <<< Grid_Image, 1 >>> (dev_imagem, canais);

	cudaMemcpy(imagem, dev_imagem, altura * largura * canais, cudaMemcpyDeviceToHost);
	cudaFree(dev_imagem);
}

__global__ void black_and_white_CUDA(unsigned char* imagem, int canais) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * canais;

	imagem[idx + 0] = imagem[idx + 0] > 127 ? 255 : 0;
	imagem[idx + 1] = imagem[idx + 1] > 127 ? 255 : 0;
	imagem[idx + 2] = imagem[idx + 2] > 127 ? 255 : 0;
}