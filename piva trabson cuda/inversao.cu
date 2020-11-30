#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Inversao.h"

__global__ void inversao_CUDA(unsigned char* imagem, int canais);

void inversao(unsigned char* imagem, int altura, int largura, int canais) {
	unsigned char* dev_imagem = NULL;

	cudaMalloc((void**)&dev_imagem, altura * largura * canais);

	cudaMemcpy(dev_imagem, imagem, altura * largura * canais, cudaMemcpyHostToDevice);

	dim3 Grid_Image(largura, altura);
	inversao_CUDA <<<Grid_Image, 1>>> (dev_imagem, canais);

	cudaMemcpy(imagem, dev_imagem, altura * largura * canais, cudaMemcpyDeviceToHost);
	cudaFree(dev_imagem);
}

__global__ void inversao_CUDA(unsigned char* imagem, int canais) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * canais;

	for (int i = 0; i < canais; i++) {
		imagem[idx + i] = 255 - imagem[idx + i];
	}
}