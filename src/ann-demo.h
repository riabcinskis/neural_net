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

    }
    void readMnist(string filename, vector<double*> &arr);
    void readMnistLabel(string filename, vector<double> &vec);
		int reverseInt(int i);
		void trainData(PictureData& pictures, string picFile, string labFile, AnnSerialDBL* SerialDBL);
		void testNet(PictureData& pictures, string picFile, string labFile, AnnSerialDBL* SerialDBL);
		void pushTarget(double a, vector<double*> &targets, ofstream &fr);
};

#endif /* ANN_DEMO_HEADER */
