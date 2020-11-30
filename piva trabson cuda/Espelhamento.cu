#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Espelhamento.h"

__global__ void espelhamento_CUDA(unsigned char* imagem, int canais);

void espelhamento(unsigned char* imagem, int altura, int largura, int canais) {
	unsigned char* dev_imagem = NULL;

	cudaMalloc((void**)&dev_imagem, altura * largura * canais);

	cudaMemcpy(dev_imagem, imagem, altura * largura * canais, cudaMemcpyHostToDevice);

	dim3 Grid_Image(largura / 2, altura);
	espelhamento_CUDA <<< Grid_Image, 1 >>> (dev_imagem, canais);

	cudaMemcpy(imagem, dev_imagem, altura * largura * canais, cudaMemcpyDeviceToHost);
	cudaFree(dev_imagem);
}

__global__ void espelhamento_CUDA(unsigned char* imagem, int canais) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * (gridDim.x * 2)) * canais;
	int idx2 = ((gridDim.x * 2) - x + y * (gridDim.x * 2)) * canais;

	for (int i = 0; i < canais; i++) {
		char aux = imagem[idx + i];
		imagem[idx + i] = imagem[idx2 + i];
		imagem[idx2 + i] = aux;
	}
}