#pragma once
class activation {
public:
	static int step(double x);
	static double sigmoid(double x);
	static double* softmax(double *x, int n);
};
