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

		void ReadData(string Mnist_file, string MnistLabel_file);
private:
    void readMnist(string filename, vector<double*> &arr);
    void readMnistLabel(string filename, vector<int> &vec);
		int reverseInt(int i);
		void pushTarget(int a, vector<double*> &targets);
};

class PictureDataFlt : public Data_Float
{
public:
    PictureDataFlt()
    {

    }

		void ReadData(string Mnist_file, string MnistLabel_file);
private:
    void readMnist(string filename, vector<float*> &arr);
    void readMnistLabel(string filename, vector<int> &vec);
		int reverseInt(int i);
		void pushTarget(int a, vector<float*> &targets);
};

#define IMPL_DOUBLE 0
#define IMPL_FLOAT 1
#define IMPL_CUDA 2

class TrainConfig{
	private:
		int mImpl;
		string mPicDataFileName;
		string mLabelDataFileName;
		int mEpochCount;
		string mErrorsFileName;
		string mNetworkFileName;
		Topology *mTopology;
		double mEta;
		double mAlpha;
	public:
		void setImpl(int impl);
		void setPicDataFileName(string picDataFileName);
		void setLabelDataFileName(string labelDataFileName);
		void setEpochCount(int epochCount);
		void setErrorsFileName(string errorsFileName);
		void setNetworkFileName(string networkFileName);
		void setTopology(Topology *topology);
		void setEta(double eta);
		void setAlpha(double alpha);

		int getImpl();
		string getPicDataFileName();
		string getLabelDataFileName();
		int getEpochCount();
		string getErrorsFileName();
		string getNetworkFileName();
		Topology* getTopology();
		double getEta();
		double getAlpha();

};

class PictureClassification{
public:
	PictureClassification() {}

	static void Train(TrainConfig *config);
	static void Test(string Mnist_file,string MnistLabel_file, string file_load_network);



private:
	static void train_network(PictureData pictures,AnnSerialDBL* serialDBL, TrainConfig *config);
	static void test_network(PictureData pictures,AnnSerialDBL* serialDBL);
	static int getMaxValue(double * a);

	static void train_network(PictureDataFlt pictures, AnnSerialFLT* serialFLT, TrainConfig *config);
	static void test_network(PictureDataFlt pictures, AnnSerialFLT* serialFLT);
	static int getMaxValue(float * a);

};

#endif /* ANN_DEMO_HEADER */
