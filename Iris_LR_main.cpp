#include <iostream>
#include <fstream>
#include <list>
#include <random>
#include "LogisticRegression.h"
using namespace std;

int patterns = 3;   // number of classes
int train_N = 120; // 40 * patterns;
int test_N = 30;   // 10 * patterns;
int nIn = 4;       // 4 features
int nOut = 3;      //
int * shuffle(int N);
void get_iris_data(double **, int **, double **, int **);
void main() {

  // Declare variables and constants
	double **train_X = new double*[train_N]; for(int i=0;i<train_N;i++) train_X[i] = new double[nIn];
	int** train_T = new int*[train_N];  for (int i = 0; i<train_N; i++) train_T[i] = new int[nOut];
	// shuffle the input data
	double **shuffled_train_X = new double*[train_N]; 
	int **shuffled_train_T = new int*[train_N];
	int * shuffled = shuffle(train_N);
	for (int i = 0; i < train_N; i++) {
		shuffled_train_X[i] = train_X[shuffled[i]];
		shuffled_train_T[i] = train_T[shuffled[i]];
	}

	double **test_X = new double*[test_N]; for (int i = 0; i<test_N; i++) test_X[i] = new double[nIn];
	int** test_T = new int*[test_N];  for (int i = 0; i<test_N; i++) test_T[i] = new int[nOut];
	int** predicted_T = new int*[test_N];  //for (int i = 0; i<test_N; i++) predicted_T[i] = new int[nOut];

	get_iris_data(train_X, train_T, test_X, test_T);

	int epochs = 2000;
	double learningRate = 0.2;

	int minibatchSize = 10;  //  number of data in each minibatch
	int minibatch_N = train_N / minibatchSize; //  number of minibatches
	//
	// Build Logistic Regression model
	//
	// construct logistic regression
	LogisticRegression classifier = LogisticRegression(nIn, nOut, minibatchSize);

	cout << "(nIn nOut mini)" << classifier.nIn << " " << classifier.nOut << " " << classifier.minibatchSize << "\n";
	/*for (int i = 0; i < nOut; i++) {
		for (int j = 0; j < nIn; j++) cout << classifier.W[i][j] << "  ";
		cout << " W  b = " << classifier.b[i] << endl;
	}*/
	
	// train
	for (int epoch = 0; epoch < epochs; epoch++) {
		for (int n = 0; n < train_N; n += minibatchSize) {
			classifier.train(shuffled_train_X+n, shuffled_train_T+n, minibatchSize, learningRate);
		}
		learningRate *= 0.95;
	}

	// test
	for (int i = 0; i < test_N; i++) {
		predicted_T[i] = classifier.predict(test_X[i]);
		/*if((i % 100)==0)*/ cout << "predicted " << " " << predicted_T[i][0] << " " << predicted_T[i][1] << " " << predicted_T[i][2] << endl;
		/*if((i % 100)==0)*/ cout << "·¹¾Ë "  << " " << test_T[i][0] << " " << test_T[i][1] << " " << test_T[i][2] << endl;
	}
	//
	// Evaluate the model
	//
	int **confusionMatrix = new int*[patterns]; for(int i=0;i<patterns;i++)confusionMatrix[i]=new int[patterns];
	double accuracy = 0.;
	double *precision = new double[patterns];
	double *recall = new double[patterns];
	for (int i = 0; i < patterns; i++) {
		precision[i] = recall[i] = 0;
		for (int j = 0; j < patterns; j++) confusionMatrix[i][j] = 0;
	}

	for (int i = 0; i < test_N; i++) {
		int predicted_,actual_;
		for (predicted_ = 0; predicted_<patterns; predicted_++) if (predicted_T[i][predicted_] == 1) break;
		for (actual_ = 0; actual_<patterns; actual_++) if (test_T[i][actual_] == 1) break;

		if (actual_ == patterns || predicted_ == patterns) {
			cout << "no decision " << predicted_ << " " << actual_ << endl;
			getchar();
			exit(123);
		}
		confusionMatrix[actual_][predicted_] += 1;
	}

	for (int i = 0; i < patterns; i++) {
		double col_ = 0.;
		double row_ = 0.;

		for (int j = 0; j < patterns; j++) {

			if (i == j) {
				accuracy += confusionMatrix[i][j];
				precision[i] += confusionMatrix[j][i];
				recall[i] += confusionMatrix[i][j];
			}

			col_ += confusionMatrix[j][i];
			row_ += confusionMatrix[i][j];
		}
		precision[i] /= col_;
		recall[i] /= row_;
	}

	accuracy /= test_N;

	cout << "------------------------------------\n";
	cout << "Logistic Regression model evaluation\n";
	cout << "------------------------------------\n";
	cout << "Accuracy: " << accuracy * 100 << "%\n";
	cout << "Precision:\n";
	for (int i = 0; i < patterns; i++) {
		cout << " class "<< i+1<< ":"  << precision[i] * 100 << "%\n";
	}
	cout << "Recall:\n";
	for (int i = 0; i < patterns; i++) {
		cout << " class " << i + 1 << ":" << recall[i] * 100 << "%\n";
	}
	getchar();
}
// return an int array of random-shuffled positions
int * shuffle(int N) {
	int *aa = new int[N];
	int i, left, j;

	for (i = 0; i < N; i++) aa[i] = -1;
	for (i = 0; i < N; i++) {
		left = rand() % (N - i);
		for (j = 0; j < N; j++) if (aa[j] == -1) {
			if (left == 0) break; else left--;
		}
		aa[j] = i;
	}
	return aa;
}
// read iris data
void get_iris_data(double **tnx, int **tnt, double **tex, int **tet) {
	ifstream iris_data("iris_data.txt");
	int it;
	for(int k = 0;k<patterns;k++){
		int m = k * 40;
		for (int i = 0; i < 40; i++,m++) {
			iris_data >> tnx[m][0] >> tnx[m][1] >> tnx[m][2] >> tnx[m][3];
			iris_data >> it;
			for (int j = 0; j < patterns; j++) tnt[m][j] = 0; tnt[m][it] = 1;
		}
		m = k * 10;
		for (int i = 0; i < 10; i++,m++) {
			iris_data >> tex[m][0] >> tex[m][1] >> tex[m][2] >> tex[m][3];
			iris_data >> it;
			for (int j = 0; j < patterns; j++) tet[m][j] = 0; tet[m][it] = 1;
		}
	}
}