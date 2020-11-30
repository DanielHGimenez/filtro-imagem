#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Grayscale.h"

__global__ void gray_scale_CUDA(unsigned char* imagem, int canais);

void gray_scale(unsigned char* imagem, int altura, int largura, int canais) {
	unsigned char* dev_imagem = NULL;

	cudaMalloc((void**)&dev_imagem, altura * largura * canais);

	cudaMemcpy(dev_imagem, imagem, altura * largura * canais, cudaMemcpyHostToDevice);

	dim3 Grid_Image(largura, altura);
	gray_scale_CUDA <<< Grid_Image, 1 >>> (dev_imagem, canais);

	cudaMemcpy(imagem, dev_imagem, altura * largura * canais, cudaMemcpyDeviceToHost);
	cudaFree(dev_imagem);
}

__global__ void gray_scale_CUDA(unsigned char* imagem, int canais) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * canais;

	char gray_color = (imagem[idx + 0] + imagem[idx + 1] + imagem[idx + 2]) / 3;

	imagem[idx + 0] = gray_color;
	imagem[idx + 1] = gray_color;
	imagem[idx + 2] = gray_color;
}