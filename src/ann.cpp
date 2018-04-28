//#include "ann.h"


// Ann.cpp : Defines the entry point for the console application.
//
#include "ann.h"

using namespace std;

//
// Random
//
Random::Random(){
  mGen = new std::mt19937();
  mDist = new std::uniform_real_distribution<double>(0., 1.);
}

double Random::next(){
  return (*mDist)(*mGen);
}

int Random::nextInt(int min, int max){
  double range = max - min;
  double r = range * next();
  return min + (int)(r+0.5);
}

bool Random::nextBool(){
  if(next() >= 0) return true;
  return false;
}


//*****************************
//
// Topology
//
Topology::Topology(){
	ml = new vector<int>();
}

Topology::~Topology(){
	ml->clear();
	delete ml;
}

void Topology::addLayer(int size){
	ml->push_back(size);
}

int Topology::getLayerCount(){
	return ml->size();
}

int Topology::getLayerSize(int index){
	return (*ml)[index];
}

int Topology::obtainNeuronCount(){
	int count = 0;
	for (int i = 0; i < ml->size(); i++)
		count += (*ml)[i] + 1;
	return count;
}

int Topology::obtainWeightCount(){
	int count = 0;
	for (int i = 0; i < ml->size()-1; i++)
		count += ((*ml)[i] + 1)*(*ml)[i+1];
	return count;
}

int Topology::getInputNeuronCount(){
	return (*ml)[0];
}

int Topology::getOutputNeuronCount(){
	return (*ml)[ml->size()-1];
}

void Topology::printTopology(FILE *file){
  int a=getLayerCount();
  fwrite (&a , sizeof(int), 1, file);
  for(int i=0;i<a;i++){
    int b=ml->at(i);
    fwrite (&b , sizeof(int), 1, file);
  }
}

void Topology::readTopology(FILE *file){
  int size=0;
  (void)fread (&size , sizeof(int), 1, file);
  int* abc=new int[size];
  ml=new vector<int>();
  (void)fread (abc , sizeof(int), size, file);
  for(int i=0;i<size;i++){
    ml->push_back(abc[i]);
  }
}


//*********
//
//Data_Double
//
double * Data_Double::getInput(int index){
	return data[index].input;
}

double * Data_Double::getOutput(int index){
	return data[index].output;
}

void Data_Double::addSample(Sample_Double sample){
	data.push_back(sample);
}

void Data_Double::setSizes(int input_size, int output_size){
	inputs = input_size;
	outputs = output_size;
}


//*********
//
//Data_Float
//
float * Data_Float::getInput(int index){
	return data[index].input;
}

float * Data_Float::getOutput(int index){
	return data[index].output;
}

void Data_Float::addSample(Sample_Float sample){
	data.push_back(sample);
}

void Data_Float::setSizes(int input_size, int output_size){
	inputs = input_size;
	outputs = output_size;
}


//***********************************
//
//AnnSerialDBL
//
void AnnSerialDBL::prepare(Topology *top){

  cTopology = top;

	l = new int[cTopology->getLayerCount()];
	s = new int[cTopology->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new double[neuronCount];
	z_arr = new double[neuronCount];

	W = new int[cTopology->getLayerCount()];
	sw = new int[cTopology->getLayerCount()];

	w_arr = new double[weightCount];
	dw_arr = new double[weightCount];

	t_arr = new double[cTopology->getLayerSize(cTopology->getLayerCount() - 1)];

	gjl = new double[neuronCount];
}

void AnnSerialDBL::init(FILE * pFile=NULL){
  L = cTopology->getLayerCount();

	Random *rnd = new Random();

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = cTopology->getLayerSize(i) + 1;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_arr[s[i + 1] - 1] = 1;
	}


	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {
		W[i] = l[i] * (l[i + 1] - 1);
		sw[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw[i] += W[j];
			}
		}
  }


  if (pFile==NULL) {
    for (int i = 0; i < L - 1; i++)
      for (int j = 0; j < W[i]; j++) {
        w_arr[sw[i] + j] =(rnd->next()*2-1); // (double)rand() / double(RAND_MAX);
        dw_arr[sw[i] + j] = 0.0;
    }
  }
  else {
    readf_Network(pFile);
  }

}

void AnnSerialDBL::train(double *a, double *b, double alpha, double eta){


	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();


	for (int i = 0; i < cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++) {
		t_arr[i] = b[i];
	}
	calc_gjl();

	//back propogation:
	for (int i = 0; i <L - 1; i++) {//per sluoksnius
		for (int j = 0; j < l[i]; j++) {//per neuronus
			for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
				dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k), dw_arr[sw[i] + k + j*(l[i + 1] - 1)], alpha, eta);
				w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
			}
		}
	}


}

void AnnSerialDBL::feedForward(double *a, double *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
		b[i] = a_arr[s[L - 1] + i];
}

void AnnSerialDBL::calc_feedForward(){
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
		for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
			for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
				z_arr[s[i + 1] + k] += w_arr[sw[i] + k + j*(l[i + 1] - 1)] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
		}
	}
}

void AnnSerialDBL::calc_gjl(){
	for (int i = L - 1; i >= 0; i--) {
		for (int j = 0; j < l[i]-1; j++) {
			if (L - 1 == i) {
				gjl[s[i] + j] = gL(a_arr[s[i] + j], z_arr[s[i] + j], t_arr[j]);
			}
			else {
				gjl[s[i] + j] = f_deriv(z_arr[s[i] + j]);
				double sum = 0;
				for (int k = 0; k < l[i + 1] - 1; k++) {
					sum += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * gjl[s[i + 1] + k];
				}
				gjl[s[i] + j] *= sum;
			}
		}
	}
}

double AnnSerialDBL::delta_w(double grad, double dw, double alpha, double eta) {
	return -eta*grad + alpha*dw;
}

double AnnSerialDBL::gL(double a, double z, double t) {
	double w = f_deriv(z) * (a - t);
	return w;
}

double AnnSerialDBL::f(double x) {
	//return atan(x)/M_PI + 0.5;
	double y = 1 + exp(-x);
	return 1 / y;
}

double AnnSerialDBL::f_deriv(double x) {
	//return 1.0 / (1+x*x);
	return exp(-x) / pow((1 + exp(-x)), 2);
}

double AnnSerialDBL::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}

double AnnSerialDBL::obtainError(double *b){
	double error = 0;

	for(int i = 0; i < l[L-1] - 1; i++){
		//printf("%f\t %.15e\n", b[i], a_arr[s[L-1] + i]);

		double tmp = b[i] - a_arr[s[L-1] + i];
		error += tmp*tmp;
	}
	return error;
}

void AnnSerialDBL::destroy(){
	delete[] l;
	l = NULL;
	delete[] s;
	s = NULL;

	delete[] a_arr;
	a_arr = NULL;
	delete[] z_arr;
	z_arr = NULL;

	delete[] W;
	W = NULL;
	delete[] sw;
	sw = NULL;

	delete[] w_arr;
	w_arr = NULL;
	delete[] dw_arr;
	dw_arr = NULL;

	delete[] t_arr;
	t_arr = NULL;

	delete[] gjl;
	gjl = NULL;
}

double* AnnSerialDBL::getWeights(){
	return w_arr;
}

double* AnnSerialDBL::getDWeights(){
	return dw_arr;
}

double* AnnSerialDBL::getA(){
	return a_arr;
}

Topology* AnnSerialDBL::getTopology(){
  return cTopology;
}

void AnnSerialDBL::print_out(){
	printf("z = %e\n", z_arr[s[L-1]+0]);
	printf("g = %e\n", gjl[s[L-1]+0]);

	for(int i = 0; i < l[L-2]; i++){
		if(i < l[L-2]) printf("[%d] z=%e, a=%e, w=%e, grad = %e\n", i, z_arr[s[L-2]+i], a_arr[s[L-2]+i], w_arr[sw[L-2] + i*(l[L-1]-1)], a_arr[s[L-2]+i]*gjl[s[L-1]+0]);
	}
}

void AnnSerialDBL::printf_Network(string output_filename){
  FILE * pFile;
  const char * c = output_filename.c_str();
  pFile = fopen(c, "wb");
  cTopology->printTopology(pFile);
  fwrite (w_arr , sizeof(double), cTopology->obtainWeightCount(), pFile);
  fwrite (dw_arr , sizeof(double), cTopology->obtainWeightCount(), pFile);
  fclose (pFile);
}

void AnnSerialDBL::readf_Network(FILE *pFile){
 (void)fread (w_arr , sizeof(double), cTopology->obtainWeightCount(), pFile);
 (void)fread (dw_arr , sizeof(double), cTopology->obtainWeightCount(), pFile);
}



//*************
//
//AnnSerialFLT
//
void AnnSerialFLT::prepare( Topology *top){
	cTopology = top;

	l = new int[top->getLayerCount()];
	s = new int[top->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	a_arr = new float[neuronCount];
	z_arr = new float[neuronCount];

	W = new int[top->getLayerCount()];
	sw = new int[top->getLayerCount()];

	w_arr = new float[weightCount];
	dw_arr = new float[weightCount];

	t_arr = new float[top->getLayerSize(top->getLayerCount() - 1)];

	gjl = new float[neuronCount];
}

void AnnSerialFLT::init(FILE *pFile=NULL){
  L = cTopology->getLayerCount();

	Random *rnd = new Random();

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		l[i] = cTopology->getLayerSize(i) + 1;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_arr[s[i + 1] - 1] = 1;
	}


	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {
		W[i] = l[i] * (l[i + 1] - 1);
		sw[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw[i] += W[j];
			}
		}
  }

  for (int i = 0; i < L - 1; i++)
    for (int j = 0; j < W[i]; j++) {
      w_arr[sw[i] + j] =(rnd->next()*2-1); // (double)rand() / double(RAND_MAX);
      dw_arr[sw[i] + j] = 0.0;
  }
}

void AnnSerialFLT::train(float *a, float *b, float alpha, float eta){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();


	for (int i = 0; i < cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++) {
		t_arr[i] = b[i];
	}
	calc_gjl();

	//back propogation:
	for (int i = 0; i <L - 1; i++) {//per sluoksnius
		for (int j = 0; j < l[i]; j++) {//per neuronus
			for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
				dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k), dw_arr[sw[i] + k + j*(l[i + 1] - 1)], alpha, eta);
				w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
			}
		}
	}
}

void AnnSerialFLT::feedForward(float *a, float *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}


	calc_feedForward();


	for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++)
		b[i] = a_arr[s[L - 1] + i];
}

void AnnSerialFLT::calc_feedForward(){
	for (int i = 0; i < L - 1; i++) {//per sluoksnius einu+
		for (int j = 0; j < l[i]; j++) { //kiek neuronu sluoksnyje+
			for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z+
				z_arr[s[i + 1] + k] += w_arr[sw[i] + k + j*(l[i + 1] - 1)] * a_arr[s[i] + j];
			}
		}
		for (int k = 0; k < l[i + 1] - 1; k++) {//per sekancio sluoksnio z
			a_arr[s[i + 1] + k] = f(z_arr[s[i + 1] + k]);
		}
	}
}

void AnnSerialFLT::calc_gjl(){
	for (int i = L - 1; i >= 0; i--) {
		for (int j = 0; j < l[i]-1; j++) {
			if (L - 1 == i) {
				gjl[s[i] + j] = gL(a_arr[s[i] + j], z_arr[s[i] + j], t_arr[j]);
			}
			else {
				gjl[s[i] + j] = f_deriv(z_arr[s[i] + j]);
				float sum = 0;
				for (int k = 0; k < l[i + 1] - 1; k++) {
					sum += w_arr[sw[i] + j*(l[i + 1] - 1) + k] * gjl[s[i + 1] + k];
				}
				gjl[s[i] + j] *= sum;
			}
		}
	}
}

float AnnSerialFLT::delta_w(float grad, float dw, float alpha, float eta) {
	return -eta*grad + alpha*dw;
}

float AnnSerialFLT::gL(float a, float z, float t) {
	float w = f_deriv(z) * (a - t);
	return w;
}

float AnnSerialFLT::f(float x) {
		//return atanf(x)/M_PI + 0.5;
	float y = 1 + exp(-x);
	return 1 / y;
}

float AnnSerialFLT::f_deriv(float x) {
	//return  1.0 / (1.0+ x*x);
	 return exp(-x) / pow((1 + exp(-x)), 2);
}

float AnnSerialFLT::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}

float AnnSerialFLT::obtainError(float *b){
	float error = 0;
	for(int i = 0; i < l[L-1] - 1; i++){
		float tmp = b[i] - a_arr[s[L-1] + i];
		error += tmp*tmp;
	}
	return error;
}

void AnnSerialFLT::destroy(){
	delete[] l;
	l = NULL;
	delete[] s;
	s = NULL;

	delete[] a_arr;
	a_arr = NULL;
	delete[] z_arr;
	z_arr = NULL;

	delete[] W;
	W = NULL;
	delete[] sw;
	sw = NULL;

	delete[] w_arr;
	w_arr = NULL;
	delete[] dw_arr;
	dw_arr = NULL;

	delete[] t_arr;
	t_arr = NULL;

	delete[] gjl;
	gjl = NULL;
}

float* AnnSerialFLT::getWeights(){
	return w_arr;
}

void AnnSerialFLT::print_out(){
  printf("z = %e\n", z_arr[s[L-1]+0]);
	printf("g = %e\n", gjl[s[L-1]+0]);

	for(int i = 0; i < l[L-2]; i++){
		if(i < l[L-2]) printf("[%d] z=%e, a=%e, w=%e, grad = %e\n", i, z_arr[s[L-2]+i], a_arr[s[L-2]+i], w_arr[sw[L-2] + i*(l[L-1]-1)], a_arr[s[L-2]+i]*gjl[s[L-1]+0]);
	}
}

void AnnSerialFLT::printf_Network(string filename){
  FILE * pFile;
  const char * c = filename.c_str();
  pFile = fopen(c, "wb");
  cTopology->printTopology(pFile);

  int weightCount = cTopology->obtainWeightCount();

  double *w_arr_dbl = new double[weightCount];
  double *dw_arr_dbl = new double[weightCount];
  for(int i = 0; i < weightCount; i++){
    w_arr_dbl[i] = (double)w_arr[i];
    dw_arr_dbl[i] = (double)dw_arr[i];
  }

  fwrite (w_arr_dbl , sizeof(double), weightCount, pFile);
  fwrite (dw_arr_dbl , sizeof(double), weightCount, pFile);
  fclose (pFile);
}
