#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Brightness.h"

__global__ void brightness_CUDA(unsigned char* imagem, int brightness, int canais);

void brightness(unsigned char* imagem, int brightness, int altura, int largura, int canais) {
	unsigned char* dev_imagem = NULL;

	cudaMalloc((void**)&dev_imagem, altura * largura * canais);

	cudaMemcpy(dev_imagem, imagem, altura * largura * canais, cudaMemcpyHostToDevice);

	dim3 Grid_Image(largura, altura);
	brightness_CUDA <<< Grid_Image, 1 >>> (dev_imagem, brightness, canais);

	cudaMemcpy(imagem, dev_imagem, altura * largura * canais, cudaMemcpyDeviceToHost);
	cudaFree(dev_imagem);
}

__global__ void brightness_CUDA(unsigned char* imagem, int brightness, int canais) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * canais;

	for (int i = 0; i < canais; i++) {
		int pixel = imagem[idx + i] + brightness;
		
		if (pixel > 255)
			pixel = 255;
		else if (pixel < 0)
			pixel = 0;

		imagem[idx + i] = pixel;
	}
}