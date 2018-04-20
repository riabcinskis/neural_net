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
		void Train(AnnSerialDBL* serialDBL);
		void Test(AnnSerialDBL* serialDBL);
private:
    void readMnist(string filename, vector<double*> &arr);
    void readMnistLabel(string filename, vector<double> &vec);
		int reverseInt(int i);
		void pushTarget(double a, vector<double*> &targets);
};

#endif /* ANN_DEMO_HEADER */
