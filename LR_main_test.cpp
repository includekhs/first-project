#include <iostream>
#include <fstream>
#include <list>
#include <random>
#include "LogisticRegression.h"
using namespace std;

int * shuffle(int N);
void main() {
	int patterns = 3;   // number of classes
	int train_N = 1200; // 400 * patterns;
	int test_N = 180;   // 60 * patterns;
	int nIn = 2;
	int nOut = 3; //
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

	int epochs = 2000;
	double learningRate = 0.2;

	int minibatchSize = 50;  //  number of data in each minibatch
	int minibatch_N = train_N / minibatchSize; //  number of minibatches

	/*double ***train_X_minibatch = new double**[minibatch_N];
	for (int i = 0; i < minibatch_N; i++){ // minibatches of training data
		train_X_minibatch[i] = new double*[i];
			for (int j = 0; j < minibatchSize; j++)train_X_minibatch[i][j] = new double[nIn];
	}
	int ***train_T_minibatch = new int**[minibatch_N]; // // minibatches of output data for training
	for (int i = 0; i < minibatch_N; i++) { // minibatches of training data
		train_T_minibatch[i] = new int*[i];
		for (int j = 0; j < minibatchSize; j++)train_T_minibatch[i][j] = new int[nIn];
	}
	list<int> minibatchIndex;// = new ArrayList<>();  // data index for minibatch to apply SGD
	for (int i = 0; i < train_N; i++) minibatchIndex.push_back(i);
	random_device rd;
	mt19937 gen(rd());
	shuffle(minibatchIndex.begin(), minibatchIndex.end(), gen);*/
	//Collections.shuffle(minibatchIndex, rng);  // shuffle data index for SGD
	//
	// Training data for demo
	//   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
    //   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
    //   class 3 : x3 ~ N(  0.0, 1.0 ), y3 ~ N(  0.0, 1.0 )
    //
	default_random_engine generator;
	normal_distribution<double> g1(-2.0, 1.0);
	normal_distribution<double> g2(2.0, 1.0);
	normal_distribution<double> g3(0.0, 1.0);

	ofstream suh("LRdata60X3test.txt");
	// data set in class 1
	for (int i = 0; i < train_N / patterns ; i++) {
		train_X[i][0] = g1(generator);
		train_X[i][1] = g2(generator);
		train_T[i][0] = 1; train_T[i][1] = 0; train_T[i][2] = 0;
	}
	for (int i = 0; i < test_N / patterns; i++) {
		test_X[i][0] = g1(generator);
		test_X[i][1] = g2(generator);
		test_T[i][0] = 1; test_T[i][1] = 0; test_T[i][2] = 0;
		suh << test_X[i][0] << "    " << test_X[i][1] << endl;
	}
	// data set in class 2
	for (int i = train_N / patterns; i < train_N / patterns * 2; i++) {
		train_X[i][0] = g2(generator);
		train_X[i][1] = g1(generator);
		train_T[i][0] = 0; train_T[i][1] = 1; train_T[i][2] = 0;
	}
	for (int i = test_N / patterns; i < test_N / patterns * 2; i++) {
		test_X[i][0] = g2(generator);
		test_X[i][1] = g1(generator);
		test_T[i][0] = 0; test_T[i][1] = 1; test_T[i][2] = 0;
		suh << test_X[i][0] << "    " << test_X[i][1] << endl;
	}
	// data set in class 3
	for (int i = train_N / patterns * 2; i < train_N; i++) {
		train_X[i][0] = g3(generator);
		train_X[i][1] = g3(generator);
		train_T[i][0] = 0; train_T[i][1] = 0; train_T[i][2] = 1;
	}
	for (int i = test_N / patterns * 2; i < test_N; i++) {
		test_X[i][0] = g3(generator);
		test_X[i][1] = g3(generator);
		test_T[i][0] = 0; test_T[i][1] = 0; test_T[i][2] = 1;
		suh << test_X[i][0] << "    " << test_X[i][1] << endl;
	}
	suh.close();
	/*/ create minibatches with training data
	list<int>::iterator iter;
	iter = minibatchIndex.begin();
	for (int i = 0; i < minibatch_N; i++,iter++) {
			for (int j = 0; j < minibatchSize; j++) {
			train_X_minibatch[i][j] = train_X[*iter * minibatchSize + j];
			train_T_minibatch[i][j] = train_T[*iter * minibatchSize + j];
		}
	}*/
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
			/*if (n == 550) {
				for (int i = 0; i < minibatchSize; i++) {
					cout << " X: " << shuffled_train_X[n+i][0] << " " << shuffled_train_X[n+i][1] << endl;
					cout << " T: " << shuffled_train_T[n+i][0] << " " << shuffled_train_T[n+i][1] << " n = " << n+i << endl;
				}
				//getchar();
			} //getchar();*/
			classifier.train(shuffled_train_X+n, shuffled_train_T+n, minibatchSize, learningRate);
		}
		learningRate *= 0.95;
	}
	
/*	for (int i = 0; i < nOut; i++) {
		for (int j = 0; j < nIn; j++) cout << classifier.W[i][j] << "  ";
		cout <<  " W  b = " << classifier.b[i] <<  endl;
	}*/

	
	// test
	for (int i = 0; i < test_N; i++) {
		predicted_T[i] = classifier.predict(test_X[i]);
		if((i % 100)==0) cout << "i=" << i <<" "<< predicted_T[i][0] << " " <<predicted_T[i][1] << " "<< predicted_T[i][2] << endl;
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
	cout << "Accuracy: " << accuracy * 100 << "\n";
	cout << "Precision:\n";
	for (int i = 0; i < patterns; i++) {
		cout << " class "<< i+1<< ":"  << precision[i] * 100 << "\n";
	}
	cout << "Recall:\n";
	for (int i = 0; i < patterns; i++) {
		cout << " class " << i + 1 << ":" << recall[i] * 100 << "\n";
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