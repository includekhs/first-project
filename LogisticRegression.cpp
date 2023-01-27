#include "LogisticRegression.h"
#include "activation.h"

void LogisticRegression::train(double **X, int **T, int minibatchSize, double learningRate) {
		// train with SGD
		// 1. calculate gradient of W, b
		for (int n = 0; n < minibatchSize; n++) {

			double *predicted_Y_ = output(X[n]);

			for (int j = 0; j < nOut; j++) {
				dY[n][j] = predicted_Y_[j] - T[n][j];

				for (int i = 0; i < nIn; i++) {
					grad_W[j][i] += dY[n][j] * X[n][i];
				}

				grad_b[j] += dY[n][j];
			}
			delete predicted_Y_;// Java --> C++
		}

		// 2. update params
		for (int j = 0; j < nOut; j++) {
			for (int i = 0; i < nIn; i++) {
				W[j][i] -= learningRate * grad_W[j][i] / minibatchSize;
			}
			b[j] -= learningRate * grad_b[j] / minibatchSize;
		}
		//return dY;
	}

	double * LogisticRegression::output(double *x) {
		activation myAct;
		double *preActivation = new double[nOut];
		for (int i = 0; i < nOut; i++) preActivation[i] = 0;

		for (int j = 0; j < nOut; j++) {

			for (int i = 0; i < nIn; i++) {
				preActivation[j] += W[j][i] * x[i];
			}

			preActivation[j] += b[j];  // linear output
		}

		return myAct.softmax(preActivation, nOut);
	}

	int*  LogisticRegression::predict(double *x) {

		double *y = output(x);  // activate input data through learned networks
		int *t = new int[nOut]; // output is the probability, so cast it to label

		int argmax = -1;
		double max = 0.;

		for (int i = 0; i < nOut; i++) {
			if (max < y[i]) {
				max = y[i];
				argmax = i;
			}
		}

		for (int i = 0; i < nOut; i++) {
			if (i == argmax) {
				t[i] = 1;
			}
			else {
				t[i] = 0;
			}
		}
		cout << "y " << y[0] << " " << y[1] << " " << y[2] << " \n";
		return t;
	}
