#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Inversao.h"
#include "Espelhamento.h"
#include "Grayscale.h"
#include "BlackAndWhite.h"
#include "Brightness.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	
	string imagem = "imagem/Imagem.png";

	Mat imagem1 = imread(imagem);
	Mat imagem2 = imread(imagem);
	Mat imagem3 = imread(imagem);
	Mat imagem4 = imread(imagem);
	Mat imagem5 = imread(imagem);

	inversao(imagem1.data, imagem1.rows, imagem1.cols, imagem1.channels());
	espelhamento(imagem2.data, imagem2.rows, imagem2.cols, imagem2.channels());
	gray_scale(imagem3.data, imagem3.rows, imagem3.cols, imagem3.channels());
	black_and_white(imagem4.data, imagem4.rows, imagem4.cols, imagem4.channels());
	brightness(imagem5.data, 30, imagem5.rows, imagem5.cols, imagem5.channels());

	imwrite("imagem/inversão.png", imagem1);
	imwrite("imagem/espelhamento.png", imagem2);
	imwrite("imagem/tonz de cinza.png", imagem3);
	imwrite("imagem/preto e branco.png", imagem4);
	imwrite("imagem/brilho.png", imagem5);

	return 0;
}