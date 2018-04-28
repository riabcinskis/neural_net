#include "ann-demo.h"

void XOR::generate(int n){
	for (int i = 0; i < n / 4; i++){
		for (double j = 0; j < 2; j++) {
			for (double k = 0; k < 2; k++) {
				double * input = new double[2];
				input[0] = j;
				input[1] = k;
				double * output = new double[2];
				output[0] = j == k;
				output[1] = j != k;
				addSample({ input,output });
			}
		}
	}
}

int XOR::getResult(int index){

		double max = 0;
		int index1 = 0;

		double* ats=getOutput(index);

		for (int i = 0; i<outputs; i++) {
		 	if (max < ats[i]) {
		 		max = ats[i];
		 		index1 = i;
		 	}
		 }
		 return index1;
}

void XOR_Float::generate(int n){
	for (int i = 0; i < n / 4; i++){
		for (float j = 0; j < 2; j++) {
			for (float k = 0; k < 2; k++) {
				float * input = new float[2];
				input[0] = j;
				input[1] = k;
				float * output = new float[2];
				output[0] = j == k;
				output[1] = j != k;
				addSample({ input,output });
			}
		}
	}
}

int XOR_Float::getResult(int index){

		float max = 0;
		int index1 = 0;

		float* ats=getOutput(index);

		for (int i = 0; i<outputs; i++) {
		 	if (max < ats[i]) {
		 		max = ats[i];
		 		index1 = i;
		 	}
		 }
		 return index1;
}

void xor_sample(){
  Topology *topology = new Topology();
	topology->addLayer(2);
	topology->addLayer(5);
	topology->addLayer(2);

	AnnSerialDBL* SerialDBL=new AnnSerialDBL();

	double alpha = 0.95;
  double eta = 0.9;
	SerialDBL -> prepare(topology);

	SerialDBL->init(NULL);

	SerialDBL->print_out();

	XOR xo;
	int dataCount=500;
	xo.generate(dataCount);

	SerialDBL->train(xo.getInput(0), xo.getOutput(0), alpha, eta);


	for (int i = 1; i < xo.getNumberOfSamples(); i++) {
		SerialDBL->train(xo.getInput(i), xo.getOutput(i), alpha, eta);
	}

	//Checking results(all combinations 0 and 1)
	double *target = new double[2];
	double *output = new double[2];


	SerialDBL->print_out();

	for (double i = 0; i < 2; i++) {
		for (double j = 0; j < 2; j++) {
			double input[] = { i ,j };
			target[0] = i==j;
			target[1] = i!=j;


			SerialDBL->feedForward(input, output);
			double error = SerialDBL->obtainError(target);
			printf("error = %e\n", error);
		}
	}

	SerialDBL->destroy();
	delete SerialDBL;
}

void xor_sample_Float(){
  Topology *topology = new Topology();
	topology->addLayer(2);
	topology->addLayer(5);
	topology->addLayer(4);
	topology->addLayer(2);



	AnnSerialFLT* serialFlt=new AnnSerialFLT();

	float alpha = 0.7;
  float eta = 0.25;
	serialFlt -> prepare(topology);

	serialFlt->init(NULL);

	XOR_Float xo;
	int dataCount=5000;
	xo.generate(dataCount);
	serialFlt->train(xo.getInput(0), xo.getOutput(0), alpha, eta);


	for (int i = 1; i < xo.getNumberOfSamples(); i++) {
		serialFlt->train(xo.getInput(i), xo.getOutput(i), alpha, eta);
	}

	//Checking results(all combinations 0 and 1)
	for (float i = 0; i < 2; i++) {
		for (float j = 0; j < 2; j++) {
			float input[] = { i ,j };
			float output[] = { 0,0 };

			serialFlt->feedForward(input, output);
			Sample_Float temp={input,output};
			xo.addSample(temp);
			printf("inout:  %.2f  %.2f\n",xo.getInput(dataCount+i*2+j)[0],xo.getInput(dataCount+i*2+j)[1] );
			printf("output: %.2f  %.2f\n",xo.getOutput(dataCount+i*2+j)[0],xo.getOutput(dataCount+i*2+j)[1] );
			printf("Result: %d\n", xo.getResult(dataCount+i*2+j));
			printf("---------------------------------\n");
		}
	}

	serialFlt->destroy();
	delete serialFlt;
}


//************************************************************************
//                           Paveiksliukai
//************************************************************************

//
// PictureData
//

void PictureData::ReadData(string Mnist_file, string MnistLabel_file){
	vector<double*> arr;
	vector<int> vec;
	vector<double*> targets;
	readMnist(Mnist_file, arr);
	readMnistLabel(MnistLabel_file, vec);

	for (int i = 0; i < vec.size(); i++) {
			pushTarget(vec[i], targets);
			Sample_Double sample = {arr[i], targets[i]};
			addSample(sample);
	}
	targets.clear();
	arr.clear();
	vec.clear();
}

//Nuskaito inputus is duomenu failu
void PictureData::readMnist(string filename, vector<double*> &arr){
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
				inputs = n_rows*n_cols;

        for(int i = 0; i < number_of_images; ++i){
            double* tp = new double[inputs];
            int index;
            for(int r = 0; r < n_rows; r++){
                for(int c = 0; c < n_cols; c++){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    index = r * n_cols + c;
                    tp[index] = ((double)temp / 255 * 1.0);
                }
            }
            arr.push_back(tp);
        }
    }else{
			printf("*** failed to open file \'%s\'\n", filename.c_str());
		}
}

//Nuskaito labelius is failo
void PictureData::readMnistLabel(string filename, vector<int> &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back((int)temp);
        }
    }else{
			printf("*** failed to open file \'%s\'\n", filename.c_str());
		}
}

void PictureData::pushTarget(int a, vector<double*> &targets){
	double* temp = new double[10];

	memset(temp, 0, 10*sizeof(double));
	temp[a] = 1;

	targets.push_back(temp);
}

//Paveiksliuku nuskaitymui
int PictureData::reverseInt(int i){
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i&255;
  ch2 = (i >> 8)&255;
  ch3 = (i >> 16)&255;
  ch4 = (i >> 24)&255;
  return ((int)ch1<<24) + ((int)ch2<<16) + ((int)ch3<<8) + ch4;
}


//
// PictureDataFlt
//
void PictureDataFlt::ReadData(string Mnist_file, string MnistLabel_file){
	vector<float*> arr;
	vector<int> vec;
	vector<float*> targets;
	readMnist(Mnist_file, arr);
	readMnistLabel(MnistLabel_file, vec);

	for (int i = 0; i < vec.size(); i++) {
			pushTarget(vec[i], targets);
			Sample_Float sample = {arr[i], targets[i]};
			addSample(sample);
	}
	targets.clear();
	arr.clear();
	vec.clear();
}

//Nuskaito inputus is duomenu failu
void PictureDataFlt::readMnist(string filename, vector<float*> &arr){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
				inputs = n_rows*n_cols;

        for(int i = 0; i < number_of_images; ++i){
            float* tp = new float[inputs];
            int index;
            for(int r = 0; r < n_rows; r++){
                for(int c = 0; c < n_cols; c++){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    index = r * n_cols + c;
                    tp[index] = ((float)(temp / 255.));
                }
            }
            arr.push_back(tp);
        }
    }else{
			printf("*** failed to open file \'%s\'\n", filename.c_str());
		}
}

//Nuskaito labelius is failo
void PictureDataFlt::readMnistLabel(string filename, vector<int> &vec){
    ifstream file (filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back((int)temp);
        }
    }else{
			printf("*** failed to open file \'%s\'\n", filename.c_str());
		}
}

void PictureDataFlt::pushTarget(int a, vector<float*> &targets){
	float* temp = new float[10];

	memset(temp, 0, 10*sizeof(float));
	temp[a] = 1;

	targets.push_back(temp);
}

//Paveiksliuku nuskaitymui
int PictureDataFlt::reverseInt(int i){
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i&255;
  ch2 = (i >> 8)&255;
  ch3 = (i >> 16)&255;
  ch4 = (i >> 24)&255;
  return ((int)ch1<<24) + ((int)ch2<<16) + ((int)ch3<<8) + ch4;
}


void pic_sample() {
	string test_labels = "./../files/t10k-labels.idx1-ubyte";
  string test_images = "./../files/t10k-images.idx3-ubyte";



	Topology* topology = new Topology();
	topology->addLayer(784);
	topology->addLayer(300);
	topology->addLayer(10);


	TrainConfig *config = new TrainConfig();
	config->setImpl(IMPL_DOUBLE);
	config->setPicDataFileName("./../files/train-images.idx3-ubyte");
	config->setLabelDataFileName("./../files/train-labels.idx1-ubyte");
	config->setErrorsFileName("avg_max_error.txt");
	config->setNetworkFileName("network_data.bin");
	config->setEpochCount(1);
	config->setTopology(topology);
	config->setEta(0.005);
	config->setAlpha(0.8);

	double startTime = clock();


  PictureClassification::Train(config);

  double endTime = clock();
  double runtime = (double)(endTime-startTime)/CLOCKS_PER_SEC;

	printf("=== DOUBLE \n");
	printf("Apmokymas uztruko: %.5f sec\n", runtime);

	PictureClassification::Test(test_images,test_labels, "network_data.bin");

	////
	config->setImpl(IMPL_FLOAT);
	config->setErrorsFileName("avg_max_error_flt.txt");
	config->setNetworkFileName("network_data_flt.bin");
	config->setEpochCount(1);
	config->setTopology(topology);
	config->setEta(0.005);
	config->setAlpha(0.8);

	startTime = clock();


  PictureClassification::Train(config);

  endTime = clock();
  runtime = (double)(endTime-startTime)/CLOCKS_PER_SEC;

	printf("=== FLOAT \n");
	printf("Apmokymas uztruko: %.5f sec\n", runtime);

	PictureClassification::Test(test_images,test_labels, "network_data_flt.bin");


}

//
// TrainConfig
//

void TrainConfig::setImpl(int impl){
	mImpl = impl;
}

void TrainConfig::setPicDataFileName(string picDataFileName){
	mPicDataFileName = picDataFileName;
}

void TrainConfig::setLabelDataFileName(string labelDataFileName){
	mLabelDataFileName = labelDataFileName;
}

void TrainConfig::setEpochCount(int epochCount){
	mEpochCount = epochCount;
}

void TrainConfig::setErrorsFileName(string errorsFileName){
	mErrorsFileName = errorsFileName;
}

void TrainConfig::setNetworkFileName(string networkFileName){
	mNetworkFileName = networkFileName;
}

void TrainConfig::setTopology(Topology *topology){
	mTopology = topology;
}

void TrainConfig::setEta(double eta){
	mEta = eta;
}

void TrainConfig::setAlpha(double alpha){
	mAlpha = alpha;
}

int TrainConfig::getImpl(){
	return mImpl;
}

string TrainConfig::getPicDataFileName(){
	return mPicDataFileName;
}

string TrainConfig::getLabelDataFileName(){
	return mLabelDataFileName;
}

int TrainConfig::getEpochCount(){
	return mEpochCount;
}

string TrainConfig::getErrorsFileName(){
	return mErrorsFileName;
}

string TrainConfig::getNetworkFileName(){
	return mNetworkFileName;
}

Topology* TrainConfig::getTopology(){
	return mTopology;
}

double TrainConfig::getEta(){
	return mEta;
}

double TrainConfig::getAlpha(){
	return mAlpha;
}

void PictureClassification::Train(TrainConfig *config){
	Topology *topology = config->getTopology();

	if(config->getImpl() == IMPL_DOUBLE){
		AnnSerialDBL* serialDBL=new AnnSerialDBL();
		double alpha = config->getAlpha();
	  double eta = config->getEta();

		serialDBL -> prepare(topology);

		serialDBL->init(NULL);

		PictureData pictures;

		pictures.ReadData(config->getPicDataFileName(), config->getLabelDataFileName());


		PictureClassification::train_network(pictures, serialDBL, config);
		serialDBL->printf_Network(config->getNetworkFileName());
		//serialDBL->print_out();


		delete serialDBL;
	}

	if(config->getImpl() == IMPL_FLOAT){
		AnnSerialFLT* serialFLT=new AnnSerialFLT();
		float alpha = (float)config->getAlpha();
	  float eta = (float)config->getEta();

		serialFLT -> prepare(topology);

		serialFLT->init(NULL);

		PictureDataFlt pictures;

		pictures.ReadData(config->getPicDataFileName(), config->getLabelDataFileName());


		PictureClassification::train_network(pictures, serialFLT, config);

		serialFLT->printf_Network(config->getNetworkFileName());
		//serialFLT->print_out();

		delete serialFLT;
	}
}

void PictureClassification::train_network(PictureData pictures,AnnSerialDBL* serialDBL, TrainConfig *config){
	double alpha = config->getAlpha();
	double eta = config->getEta();
	int epoch_count = config->getEpochCount();

  double *tmpArr = new double[10];
	double *epoch_error=new double[epoch_count];
	double *max_epoch_error=new double[epoch_count];

  for (int j = 0; j < epoch_count; j++){
    for (int i = 0; i < pictures.getNumberOfSamples(); i++) {

      serialDBL->train( pictures.getInput(i),  pictures.getOutput(i), alpha, eta);

			double error = serialDBL->obtainError( pictures.getOutput(i));

			epoch_error[j]+=error;

			if(max_epoch_error[j]<error){
				max_epoch_error[j]=error;
			}
		}
		printf("+\n");
		printf("%d epocha\tavg:%.10f\tmax:%.10f\n",j+1,epoch_error[j]/pictures.getNumberOfSamples(),max_epoch_error[j]);
	}


	FILE *file = fopen(config->getErrorsFileName().c_str(), "w");
	if(file == NULL){
		printf("*** failed to open file \'\%s'\n", config->getErrorsFileName().c_str());
	}

	for(int i=0;i<epoch_count;i++){
			fprintf(file, "%d\t%.10f\t%.10f\n",i+1,epoch_error[i]/pictures.getNumberOfSamples(),max_epoch_error[i]);
	}

	fclose(file);

	delete[] tmpArr;
	delete[] epoch_error;
	delete[] max_epoch_error;
}

void PictureClassification::train_network(PictureDataFlt pictures, AnnSerialFLT* serialFLT, TrainConfig *config){

	float alpha = (float) config->getAlpha();
	float eta = (float) config->getEta();
	int epoch_count = config->getEpochCount();

	float *tmpArr = new float[10];

	float *epoch_error=new float[epoch_count];
	float *max_epoch_error=new float[epoch_count];


  for (int j = 0; j < epoch_count; j++){
    for (int i = 0; i < pictures.getNumberOfSamples(); i++) {

      serialFLT->train( pictures.getInput(i),  pictures.getOutput(i), alpha, eta);

			float error = serialFLT->obtainError( pictures.getOutput(i));

			epoch_error[j]+=error;

			if(max_epoch_error[j]<error){
				max_epoch_error[j]=error;
			}
		}
		printf("+\n");
		printf("%d epocha\tavg:%.10f\tmax:%.10f\n",j+1,epoch_error[j]/pictures.getNumberOfSamples(),max_epoch_error[j]);
	}

	FILE *file = fopen(config->getErrorsFileName().c_str(), "w");
	if(file == NULL){
		printf("*** failed to open file \'\%s'\n", config->getErrorsFileName().c_str());
	}

	for(int i=0;i<epoch_count;i++){
			fprintf(file, "%d\t%.10f\t%.10f\n",i+1,epoch_error[i]/pictures.getNumberOfSamples(),max_epoch_error[i]);
	}

	fclose(file);

	delete[] tmpArr;
	delete[] epoch_error;
	delete[] max_epoch_error;
}

void PictureClassification::Test(string Mnist_file,string MnistLabel_file, string file_load_network){
  AnnSerialDBL* test_serialDBL=new AnnSerialDBL(file_load_network);


	test_serialDBL-> prepare(NULL);

  test_serialDBL->init(NULL);

	PictureData pictures;

	pictures.ReadData(Mnist_file,MnistLabel_file);


  PictureClassification::test_network(pictures,test_serialDBL);

	delete test_serialDBL;
}

void PictureClassification::test_network(PictureData pictures,AnnSerialDBL* serialDBL){
	double *tmpArr = new double[10];

	int correct_outputs=0;
	int test_samples=pictures.getNumberOfSamples();
	for (int i = 0; i < test_samples; i++) {
		serialDBL->feedForward(pictures.getInput(i), tmpArr);


		if(pictures.getOutput(i)[PictureClassification::getMaxValue(tmpArr)]==1)
			correct_outputs++;

	}

	printf("Tests done: %d\nCorrect outputs: %d\n", test_samples,correct_outputs);

	delete[] tmpArr;
}

void PictureClassification::test_network(PictureDataFlt pictures, AnnSerialFLT* serialFLT){
	float *tmpArr = new float[10];

	int correct_outputs = 0;
	int test_samples=pictures.getNumberOfSamples();
	for (int i = 0; i < test_samples; i++) {
		serialFLT->feedForward(pictures.getInput(i), tmpArr);


		if(pictures.getOutput(i)[PictureClassification::getMaxValue(tmpArr)]==1)
			correct_outputs++;

	}

	printf("Tests done: %d\nCorrect outputs: %d\n", test_samples, correct_outputs);

	delete[] tmpArr;
}



int PictureClassification::getMaxValue(double * a) {
 int ind = 0;
 double max = 0;
 for (int i = 0; i < 10; i++) {
     if (a[i] > max) {
         max = a[i];
         ind = i;
     }
 }
 return ind;
}

int PictureClassification::getMaxValue(float * a) {
 int ind = 0;
 float max = 0;
 for (int i = 0; i < 10; i++) {
     if (a[i] > max) {
         max = a[i];
         ind = i;
     }
 }
 return ind;
}

int main (int c, char *v[]) {

  printf("ANN - demo\n\n");

 if(run_tests() == false) return 0;

//  xor_sample();

	// printf("\n\n\nFloat rezultatai: \n");
	// xor_sample_Float();
  //
   //run_cuda_sample();

	pic_sample();

 return 0;
}
