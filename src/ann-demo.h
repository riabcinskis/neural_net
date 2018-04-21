#ifndef ANN_DEMO_HEADER
#define ANN_DEMO_HEADER

#include <stdio.h>
#include <iostream>
#include "ann.h"
#include "tests.h"

using namespace std;

class XOR : public Data_Double
{
public:
	void generate(int n);

	int getResult(int index);

	XOR()
	{
		inputs = 2;
		outputs = 2;

	}
	void printInputs(int index);

	void printOutputs(int index);
};

class XOR_Float : public Data_Float
{
public:
	void generate(int n);

	int getResult(int index);

	XOR_Float()
	{
		inputs = 2;
		outputs = 2;
		samples = 0;
	}
	void printInputs(int index);

	void printOutputs(int index);
};

//************************************************************************
//                           Paveiksliukai
//************************************************************************

class PictureData : public Data_Double
{
public:
    PictureData()
    {
			inputs = 784;
			outputs = 10;
    }

		void ReadData(string Mnist_file, string MnistLabel_file);
private:
    void readMnist(string filename, vector<double*> &arr);
    void readMnistLabel(string filename, vector<double> &vec);
		int reverseInt(int i);
		void pushTarget(double a, vector<double*> &targets);
};

class PictureClassification{
public:
	PictureClassification() {}

	static void Train(string Mnist_file,string MnistLabel_file, int epoch_count,string file_avg_max_error,string file_save_network);
	static void Test(string Mnist_file,string MnistLabel_file, string file_load_network);
private:
	static void train_network(PictureData pictures,AnnSerialDBL* serialDBL, int epoch_count,string file_avg_max_error);
	static void test_network(PictureData pictures,AnnSerialDBL* serialDBL);
	static int getMaxValue(double * a);
};

#endif /* ANN_DEMO_HEADER */
