#include "ann.h"

namespace ann {


	__global__ void
	kernel(int n, float *arr){

		volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;
		if(idx >= n) return;

	        arr[idx] *= 2.0f;
	}

	__global__ void
	kernel_feedforward(
		int layer_id,
		int *l,
		int *s,
		int *sw,
		float *z_arr,
		float *a_arr,
		float *w_arr
	 ){
		volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

		int neuron_count = l[layer_id];
		int neuron_count_prev = l[layer_id-1];

		//printf("layer = %d idx = %d count = %d\n", layer_id, idx, neuron_count-1);
		if(idx >= neuron_count-1) return;

		float z = 0;
		for(int k = 0; k < neuron_count_prev; k++){
			z += w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k];
			// printf("w_arr[%d] * a_arr[%d] = %.20f\n",
			// 		sw[layer_id-1] + k*(neuron_count - 1) + idx ,
			// 		s[layer_id-1] + k,
			// 		w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k]);
			// printf("%.10f * %.10f = %.10f\n", w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx ],
			// 		a_arr[s[layer_id-1] + k],
			// 		w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k]
			// 	);

		}

		z_arr[s[layer_id] + idx] = z;
		float a = 1.0 / (1.0 + expf(-z));
		a_arr[s[layer_id] + idx] = a;
	  // printf("index = %d z = %.5f\n", s[layer_id] + idx, z);
		// printf("a = %.20f\n", a);
	}

	__global__ void
	kernel_calc_gL(
		int layer_id,
		int *l,
		int *s,
		float *z_arr,
		float *a_arr,
		float *t_arr,
		float *gjl
	 ){

		volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

		int neuron_count = l[layer_id];

		if(idx >= neuron_count-1) return;

		float z = z_arr[s[layer_id] + idx];
		float tmp = 1 + expf(-z);
		float f_deriv=expf(-z) / (tmp*tmp);

		gjl[s[layer_id] + idx] = f_deriv*(a_arr[s[layer_id] + idx] - t_arr[idx]);
	}

	__global__ void
	kernel_calc_gjL(
		int layer_id,
		int *l,
		int *s,
		int *sw,
		float *z_arr,
		float *a_arr,
		float *t_arr,
		float *gjl,
		float *w_arr
	 ){

		volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

		int neuron_count = l[layer_id];
		int neuron_count_next = l[layer_id+1];

		if(idx >= neuron_count-1) return;

		//float f_deriv=expf(-z_arr[s[layer_id] + idx]) / powf((1 + expf(-z_arr[s[layer_id] + idx])),2.0f);
		float z = z_arr[s[layer_id] + idx];
		float tmp = 1 + expf(-z);
		float f_deriv=expf(-z) / (tmp*tmp);


		float sum = 0;
		for (int k = 0; k < neuron_count_next-1; k++) {
				sum += w_arr[sw[layer_id] + idx*(l[layer_id + 1] - 1) + k] * gjl[s[layer_id + 1] + k];
		}

		gjl[s[layer_id] + idx] = f_deriv*sum;
		// printf("Kernelis %d - %.20f\n", s[layer_id] + idx, gjl[s[layer_id] + idx]);
	}


	__global__ void
	kernel_weight_update(
		int layer_id,
		int *l,
		int *s,
		int *sw,
		float *z_arr,
		float *a_arr,
		float *t_arr,
		float *gjl,
		float *w_arr,
		float *dw_arr,
		float eta,
		float alpha
	 ){

		 volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

		 int neuron_count = l[layer_id];
		 int neuron_count_next = l[layer_id+1];

		 if(idx >= neuron_count) return;

		 float a = a_arr[s[layer_id] + idx];
		 for(int k = 0; k < neuron_count_next-1; k++){

			 float grad=/*a_arr[s[layer_id] + idx]*/a*gjl[s[layer_id + 1] + k];

			 dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k]=
			 		-eta*grad+
			 		alpha*dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k];

			 w_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k]+=
			 		dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k];
		 }
	}

}

void run_cuda_sample(){

  int deviceCount = 0;
  checkCudaErrors( cudaGetDeviceCount(&deviceCount));
  if(deviceCount == 0){
    printf("*** there is no CUDE device\n");
    return;
  }

  checkCudaErrors( cudaSetDevice(0) );

  int n = 11; // number of elements

  float *arr = new float[n];
  for(int i = 0; i < n; i++)
    arr[i] = i;

  int h = 4; // number of threads in block
  int g = (n + (h-n%h))/h; // number of grids

  printf("n=%d, h=%d, g=%d\n", n, h, g);


  int bc_arr = sizeof(float)*n;

  float *dv_arr = NULL;

  checkCudaErrors( cudaMalloc((void **)&dv_arr, bc_arr) );

  checkCudaErrors( cudaMemcpy(dv_arr, arr, bc_arr, cudaMemcpyHostToDevice) );

  dim3 grid_dim(g, 1, 1);
  dim3 block_dim(h, 1, 1);

  ann::kernel<<<grid_dim, block_dim>>>(n, dv_arr);



  checkCudaErrors( cudaMemcpy(arr, dv_arr, bc_arr, cudaMemcpyDeviceToHost) );

  for(int i = 0; i < n; i++)
    printf("[%d] = %f\n", i, arr[i]);

  checkCudaErrors( cudaFree(dv_arr) );

  checkCudaErrors(cudaDeviceReset());

}



//
//AnnSerialFLT
//
void AnnCUDA::prepare( Topology *top){
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

	// cuda

	int deviceCount = 0;
	checkCudaErrors( cudaGetDeviceCount(&deviceCount));
	if(deviceCount == 0){
		printf("*** there is no CUDE device\n");
		return;
	}

	checkCudaErrors( cudaSetDevice(0) );

	dv_l = NULL; bc_l = sizeof(int)*top->getLayerCount();
	dv_s = NULL; bc_s = sizeof(int)*top->getLayerCount();;

	dv_a_arr = NULL; bc_a_arr = sizeof(float)*neuronCount;
	dv_z_arr = NULL; bc_z_arr = sizeof(float)*neuronCount;

	dv_W = NULL; bc_W = sizeof(int)*top->getLayerCount();
	dv_sw = NULL; bc_sw = sizeof(int)*top->getLayerCount();

	dv_w_arr = NULL; bc_w_arr = sizeof(float)*weightCount;
	dv_dw_arr = NULL; bc_dw_arr = sizeof(float)*weightCount;

	dv_t_arr = NULL; bc_t_arr = sizeof(float)*top->getLayerSize(top->getLayerCount() - 1);
	dv_gjl = NULL; bc_gjl = sizeof(float)*neuronCount;

	checkCudaErrors( cudaMalloc((void **)&dv_l, bc_l) );
	checkCudaErrors( cudaMalloc((void **)&dv_s, bc_s) );
	checkCudaErrors( cudaMalloc((void **)&dv_a_arr, bc_a_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_z_arr, bc_z_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_W, bc_W) );
	checkCudaErrors( cudaMalloc((void **)&dv_sw, bc_sw) );
	checkCudaErrors( cudaMalloc((void **)&dv_w_arr, bc_w_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_dw_arr, bc_dw_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_t_arr, bc_t_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_gjl, bc_gjl) );

}

void AnnCUDA::init(FILE *pFile=NULL){
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

	checkCudaErrors( cudaMemcpy(dv_w_arr, w_arr, bc_w_arr, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_dw_arr, dw_arr, bc_dw_arr, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemcpy(dv_l, l, bc_l, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_s, s, bc_s, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_sw, sw, bc_sw, cudaMemcpyHostToDevice) );
}

void AnnCUDA::train(float *a, float *b, float alpha, float eta){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	// for (int i = 0; i < 10; i++) {
	// 	printf("a[%d] = %.10f\n", i, a_arr[i]);
	// }

	// for (int i = 0; i < 7; i++) {
	// 	printf("a[%d] = %.10f\n", i, a_arr[i]);
	// }
	// printf("\n");
	// for (int i = 0; i < 7; i++) {
	// 	printf("z[%d] = %.10f\n", i, z_arr[i]);
	// }

	for (int i = 0; i < cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++) {
		t_arr[i] = b[i];
	}

	calc_gjl();

	// //back propogation:
	// for (int i = 0; i <L - 1; i++) {//per sluoksnius
	// 	for (int j = 0; j < l[i]; j++) {//per neuronus
	// 		for (int k = 0; k < l[i + 1] - 1; k++) {//per kito sluoksnio neuronus
	// 			dw_arr[sw[i] + k + j*(l[i + 1] - 1)] = delta_w(w_gradient(i, j, k), dw_arr[sw[i] + k + j*(l[i + 1] - 1)], alpha, eta);
	// 			w_arr[sw[i] + k + j*(l[i + 1] - 1)] += dw_arr[sw[i] + k + j*(l[i + 1] - 1)];
	// 		}
	// 	}
	// }

//	checkCudaErrors( cudaMemcpy(dv_a_arr, a_arr, bc_a_arr, cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dv_gjl, gjl, bc_gjl, cudaMemcpyHostToDevice) );
	//checkCudaErrors( cudaMemcpy(dv_w_arr, w_arr, bc_w_arr, cudaMemcpyHostToDevice) );


	for (int i = 0; i < L-1; i++) {//per sluoksnius einu+

		int neuron_count = l[i];
		int h = 32; // number of threads in block
		int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
		dim3 grid_dim(g, 1, 1);
		dim3 block_dim(h, 1, 1);

		// printf("%s\n", "A masyvas");
		// for (int j = 0; j < 7; j++) {
		// 	printf("a[%d] = %.20f\n", j, a_arr[j]);
		// }

		ann::kernel_weight_update<<<grid_dim, block_dim>>>(
			i,
			dv_l,
			dv_s,
			dv_sw,
			dv_z_arr,
			dv_a_arr,
			dv_t_arr,
			dv_gjl,
			dv_w_arr,
			dv_dw_arr,
			eta,
			alpha
		);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

}

void AnnCUDA::finishTraining(){
	checkCudaErrors( cudaMemcpy(w_arr, dv_w_arr, bc_w_arr, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(dw_arr, dv_dw_arr, bc_dw_arr, cudaMemcpyDeviceToHost) );
}

void AnnCUDA::feedForward(float *a, float *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount(); j++) {
		z_arr[j] = 0;
	}


	calc_feedForward();

	checkCudaErrors( cudaMemcpy(a_arr, dv_a_arr, bc_a_arr, cudaMemcpyDeviceToHost) );

	for (int i = 0; i<cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++){
		b[i] = a_arr[s[L - 1] + i];
		//printf("b[%d] = %.10f\n", i, b[i]);
	}
}

void AnnCUDA::calc_feedForward(){

	checkCudaErrors( cudaMemcpy(dv_a_arr, a_arr, bc_a_arr, cudaMemcpyHostToDevice) );



	for (int i = 1; i < L; i++) {//per sluoksnius einu+

//	printf("current layer_id = %d\n", i);
		int neuron_count = l[i];
		int h = 32; // number of threads in block
	  int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
		dim3 grid_dim(g, 1, 1);
		dim3 block_dim(h, 1, 1);

		ann::kernel_feedforward<<<grid_dim, block_dim>>>(
			i,
			dv_l,
			dv_s,
			dv_sw,
			dv_z_arr,
			dv_a_arr,
			dv_w_arr
		);

	}
}

void AnnCUDA::calc_gjl(){

	checkCudaErrors( cudaMemcpy(dv_t_arr, t_arr, bc_t_arr, cudaMemcpyHostToDevice) );


	// int last_layer_id=cTopology->getLayerCount()-1;
	int last_layer_id=L-1;
	int neuron_count = l[last_layer_id];
	int h = 32; // number of threads in block
	int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
	dim3 grid_dim(g, 1, 1);
	dim3 block_dim(h, 1, 1);


	ann::kernel_calc_gL<<<grid_dim, block_dim>>>(
		last_layer_id,
		dv_l,
		dv_s,
		dv_z_arr,
		dv_a_arr,
		dv_t_arr,
		dv_gjl
	);

	//Cia nezinau, ar i >= 0, ar i >= 1
	for (int i = L - 2; i >= 1; i--) {
			neuron_count = l[i];
			h = 32; // number of threads in block
			g = (neuron_count + (h-neuron_count%h))/h; // number of grids
			dim3 grid_dim(g, 1, 1);
			dim3 block_dim(h, 1, 1);

			ann::kernel_calc_gjL<<<grid_dim, block_dim>>>(
				i,
				dv_l,
				dv_s,
				dv_sw,
				dv_z_arr,
				dv_a_arr,
				dv_t_arr,
				dv_gjl,
				dv_w_arr
			);
		}

//	checkCudaErrors( cudaMemcpy(gjl, dv_gjl, bc_gjl, cudaMemcpyDeviceToHost) );

}

float AnnCUDA::delta_w(float grad, float dw, float alpha, float eta) {
	return -eta*grad + alpha*dw;
}

float AnnCUDA::gL(float a, float z, float t) {
	float w = f_deriv(z) * (a - t);
	return w;
}

float AnnCUDA::f(float x) {
		//return atanf(x)/M_PI + 0.5;
	float y = 1 + exp(-x);
	return 1 / y;
}

float AnnCUDA::f_deriv(float x) {
	//return  1.0 / (1.0+ x*x);
	 return exp(-x) / pow((1 + exp(-x)), 2);
}

float AnnCUDA::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}

float AnnCUDA::obtainError(float *b){
	checkCudaErrors( cudaMemcpy(a_arr, dv_a_arr, bc_a_arr, cudaMemcpyDeviceToHost) );
	float error = 0;
	for(int i = 0; i < l[L-1] - 1; i++){
		float tmp = b[i] - a_arr[s[L-1] + i];
		error += tmp*tmp;
		//printf("a_arr[%d] = %.10f\n", s[L-1] + i, a_arr[s[L-1] + i]);
	}
	return error;
}

void AnnCUDA::destroy(){
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



	checkCudaErrors( cudaFree(dv_l) );
	checkCudaErrors( cudaFree(dv_s) );
	checkCudaErrors( cudaFree(dv_a_arr) );
	checkCudaErrors( cudaFree(dv_z_arr) );
	checkCudaErrors( cudaFree(dv_W) );
	checkCudaErrors( cudaFree(dv_sw) );
	checkCudaErrors( cudaFree(dv_w_arr) );
	checkCudaErrors( cudaFree(dv_dw_arr) );
	checkCudaErrors( cudaFree(dv_t_arr) );
	checkCudaErrors( cudaFree(dv_gjl) );


  checkCudaErrors(cudaDeviceReset());
}

float* AnnCUDA::getWeights(){
	return w_arr;
}


void AnnCUDA::print_out(){
  printf("z = %e\n", z_arr[s[L-1]+0]);
	printf("g = %e\n", gjl[s[L-1]+0]);

	for(int i = 0; i < l[L-2]; i++){
		if(i < l[L-2]) printf("[%d] z=%e, a=%e, w=%e, grad = %e\n", i, z_arr[s[L-2]+i], a_arr[s[L-2]+i], w_arr[sw[L-2] + i*(l[L-1]-1)], a_arr[s[L-2]+i]*gjl[s[L-1]+0]);
	}
}

void AnnCUDA::setWeights(float *t_w_arr) {
	w_arr=t_w_arr;
	checkCudaErrors( cudaMemcpy(dv_w_arr, w_arr, bc_w_arr, cudaMemcpyHostToDevice) );
}

void AnnCUDA::printf_Network(string filename){
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


//******************Cuda 2.0***************************************

void AnnCUDA2::prepare( Topology *top){
	cTopology = top;

	l = new int[top->getLayerCount()];
	s = new int[top->getLayerCount()];
	l_real = new int[top->getLayerCount()];

	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	int neuronCount2 = cTopology->obtainNeuronCount2();
	int weightCount2 = cTopology->obtainWeightCount2();

	// printf("neuronCount = %d\n", neuronCount);
	// printf("neuronCount2 = %d\n", neuronCount2);
	// printf("weightCount = %d\n", weightCount);
	// printf("weightCount2 = %d\n", weightCount2);

	a_arr = new float[neuronCount2];
	z_arr = new float[neuronCount2];

	W = new int[top->getLayerCount()];
	W_real = new int[top->getLayerCount()];
	sw = new int[top->getLayerCount()];
	sw_real = new int[top->getLayerCount()];

	w_arr = new float[weightCount2];
	dw_arr = new float[weightCount2];

	t_arr = new float[top->getLayerSize(top->getLayerCount() - 1)];

	gjl = new float[neuronCount2];

	// cuda

	int deviceCount = 0;
	checkCudaErrors( cudaGetDeviceCount(&deviceCount));
	if(deviceCount == 0){
		printf("*** there is no CUDE device\n");
		return;
	}

	checkCudaErrors( cudaSetDevice(0) );

	dv_l = NULL; bc_l = sizeof(int)*top->getLayerCount();
	dv_s = NULL; bc_s = sizeof(int)*top->getLayerCount();;

	dv_a_arr = NULL; bc_a_arr = sizeof(float)*neuronCount2;
	dv_z_arr = NULL; bc_z_arr = sizeof(float)*neuronCount2;

	dv_W = NULL; bc_W = sizeof(int)*top->getLayerCount();
	dv_sw = NULL; bc_sw = sizeof(int)*top->getLayerCount();

	dv_w_arr = NULL; bc_w_arr = sizeof(float)*weightCount2;
	dv_dw_arr = NULL; bc_dw_arr = sizeof(float)*weightCount2;

	dv_t_arr = NULL; bc_t_arr = sizeof(float)*top->getLayerSize(top->getLayerCount() - 1);
	dv_gjl = NULL; bc_gjl = sizeof(float)*neuronCount2;

	checkCudaErrors( cudaMalloc((void **)&dv_l, bc_l) );
	checkCudaErrors( cudaMalloc((void **)&dv_s, bc_s) );
	checkCudaErrors( cudaMalloc((void **)&dv_a_arr, bc_a_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_z_arr, bc_z_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_W, bc_W) );
	checkCudaErrors( cudaMalloc((void **)&dv_sw, bc_sw) );
	checkCudaErrors( cudaMalloc((void **)&dv_w_arr, bc_w_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_dw_arr, bc_dw_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_t_arr, bc_t_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_gjl, bc_gjl) );

}

void AnnCUDA2::init(FILE *pFile=NULL){
  L = cTopology->getLayerCount();

	Random *rnd = new Random();
	int *temp_s = new int[L - 1];

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		int neuron_count = cTopology -> getLayerSize(i) + 1;
		l[i] = neuron_count + (32 - neuron_count % 32);
		l_real[i] = neuron_count;
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s[i] = 0;
		for (int j = i; j > 0; j--) {
			s[i] += l[j - 1];
		}
		if (i == 0)
			temp_s[i] = l_real[i];
		else
			temp_s[i] = s[i] + l_real[i];
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_arr[temp_s[i] - 1] = 1;
	}

	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {
		W[i] = l_real[i] * (l_real[i + 1] - 1);
		W_real[i] = W[i];
		if (W[i] % 32 != 0) {
			W[i] += (32 - W[i] % 32);
		}
		sw[i] = 0;
		sw_real[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw[i] += W[j];
				sw_real[i] += W_real[j];

			}
		}
  }

  for (int i = 0; i < L - 1; i++)
    for (int j = 0; j < W[i]; j++) {
			if (j < W_real[i]){
      	w_arr[sw[i] + j] =(rnd->next()*2-1);
			}
			else{
				w_arr[sw[i] + j] = 0.0;
			}
      dw_arr[sw[i] + j] = 0.0;
  }

	delete[] temp_s;

	checkCudaErrors( cudaMemcpy(dv_w_arr, w_arr, bc_w_arr, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_dw_arr, dw_arr, bc_dw_arr, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemcpy(dv_l, l_real, bc_l, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_s, s, bc_s, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_sw, sw, bc_sw, cudaMemcpyHostToDevice) );
}

void AnnCUDA2::train(float *a, float *b, float alpha, float eta){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount2(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i < cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++) {
		t_arr[i] = b[i];
	}

	calc_gjl();

	for (int i = 0; i < L-1; i++) {//per sluoksnius einu+

		int neuron_count = l[i];
		int h = 32; // number of threads in block
		int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
		dim3 grid_dim(g, 1, 1);
		dim3 block_dim(h, 1, 1);

		ann::kernel_weight_update<<<grid_dim, block_dim>>>(
			i,
			dv_l,
			dv_s,
			dv_sw,
			dv_z_arr,
			dv_a_arr,
			dv_t_arr,
			dv_gjl,
			dv_w_arr,
			dv_dw_arr,
			eta,
			alpha
		);
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

}

void AnnCUDA2::finishTraining(){
	checkCudaErrors( cudaMemcpy(w_arr, dv_w_arr, bc_w_arr, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(dw_arr, dv_dw_arr, bc_dw_arr, cudaMemcpyDeviceToHost) );
}

void AnnCUDA2::feedForward(float *a, float *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_arr[i] = a[i];
	}

	for (int j = 0; j < cTopology->obtainNeuronCount2(); j++) {
		z_arr[j] = 0;
	}

	calc_feedForward();

	checkCudaErrors( cudaMemcpy(a_arr, dv_a_arr, bc_a_arr, cudaMemcpyDeviceToHost) );

	for (int i = 0; i < l_real[L - 1]; i++){
		b[i] = a_arr[s[L - 1] + i];
		//printf("b[%d] = %.10f\n", i, b[i]);
	}
}

void AnnCUDA2::calc_feedForward(){

	checkCudaErrors( cudaMemcpy(dv_a_arr, a_arr, bc_a_arr, cudaMemcpyHostToDevice) );



	for (int i = 1; i < L; i++) {//per sluoksnius einu+

	//printf("current layer_id = %d\n", i);
		int neuron_count = l_real[i];
		int h = 32; // number of threads in block
	  int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
		dim3 grid_dim(g, 1, 1);
		dim3 block_dim(h, 1, 1);

		ann::kernel_feedforward<<<grid_dim, block_dim>>>(
			i,
			dv_l,
			dv_s,
			dv_sw,
			dv_z_arr,
			dv_a_arr,
			dv_w_arr
		);
	}
}

void AnnCUDA2::calc_gjl(){

	checkCudaErrors( cudaMemcpy(dv_t_arr, t_arr, bc_t_arr, cudaMemcpyHostToDevice) );


	// int last_layer_id=cTopology->getLayerCount()-1;
	int last_layer_id=L-1;
	int neuron_count = l_real[last_layer_id];
	int h = 32; // number of threads in block
	int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
	dim3 grid_dim(g, 1, 1);
	dim3 block_dim(h, 1, 1);


	ann::kernel_calc_gL<<<grid_dim, block_dim>>>(
		last_layer_id,
		dv_l,
		dv_s,
		dv_z_arr,
		dv_a_arr,
		dv_t_arr,
		dv_gjl
	);

	//Cia nezinau, ar i >= 0, ar i >= 1
	for (int i = L - 2; i >= 1; i--) {
			neuron_count = l_real[i];
			h = 32; // number of threads in block
			g = (neuron_count + (h-neuron_count%h))/h; // number of grids
			dim3 grid_dim(g, 1, 1);
			dim3 block_dim(h, 1, 1);

			ann::kernel_calc_gjL<<<grid_dim, block_dim>>>(
				i,
				dv_l,
				dv_s,
				dv_sw,
				dv_z_arr,
				dv_a_arr,
				dv_t_arr,
				dv_gjl,
				dv_w_arr
			);
		}

//	checkCudaErrors( cudaMemcpy(gjl, dv_gjl, bc_gjl, cudaMemcpyDeviceToHost) );

}

float AnnCUDA2::delta_w(float grad, float dw, float alpha, float eta) {
	return -eta*grad + alpha*dw;
}

float AnnCUDA2::gL(float a, float z, float t) {
	float w = f_deriv(z) * (a - t);
	return w;
}

float AnnCUDA2::f(float x) {
		//return atanf(x)/M_PI + 0.5;
	float y = 1 + exp(-x);
	return 1 / y;
}

float AnnCUDA2::f_deriv(float x) {
	//return  1.0 / (1.0+ x*x);
	 return exp(-x) / pow((1 + exp(-x)), 2);
}

float AnnCUDA2::w_gradient(int layer_id, int w_i, int w_j) {
	return a_arr[s[layer_id] + w_i] * gjl[s[layer_id + 1] + w_j];
}

float AnnCUDA2::obtainError(float *b){
	checkCudaErrors( cudaMemcpy(a_arr, dv_a_arr, bc_a_arr, cudaMemcpyDeviceToHost) );
	float error = 0;
	for(int i = 0; i < l_real[L-1] - 1; i++){
		float tmp = b[i] - a_arr[s[L-1] + i];
		error += tmp*tmp;
		//printf("a_arr[%d] = %.10f\n", s[L-1] + i, a_arr[s[L-1] + i]);
	}
	return error;
}

void AnnCUDA2::destroy(){
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



	checkCudaErrors( cudaFree(dv_l) );
	checkCudaErrors( cudaFree(dv_s) );
	checkCudaErrors( cudaFree(dv_a_arr) );
	checkCudaErrors( cudaFree(dv_z_arr) );
	checkCudaErrors( cudaFree(dv_W) );
	checkCudaErrors( cudaFree(dv_sw) );
	checkCudaErrors( cudaFree(dv_w_arr) );
	checkCudaErrors( cudaFree(dv_dw_arr) );
	checkCudaErrors( cudaFree(dv_t_arr) );
	checkCudaErrors( cudaFree(dv_gjl) );


  checkCudaErrors(cudaDeviceReset());
}

float* AnnCUDA2::getWeights(){
	return w_arr;
}

float* AnnCUDA2::getA(){
	return a_arr;
}

void AnnCUDA2::print_out(){
  printf("z = %e\n", z_arr[s[L-1]+0]);
	printf("g = %e\n", gjl[s[L-1]+0]);

	for(int i = 0; i < l[L-2]; i++){
		if(i < l[L-2]) printf("[%d] z=%e, a=%e, w=%e, grad = %e\n", i, z_arr[s[L-2]+i], a_arr[s[L-2]+i], w_arr[sw[L-2] + i*(l[L-1]-1)], a_arr[s[L-2]+i]*gjl[s[L-1]+0]);
	}
}

void AnnCUDA2::setWeights(float *t_w_arr) {
	int prev_count = 0;
	for (int i = 0; i < cTopology->getLayerCount() - 1; i++) {
		if (i > 0)
			prev_count += W_real[i - 1];
		for (int j = 0; j < W_real[i]; j++) {
			int index_w = sw[i] + j;
			int index_t = prev_count + j;
			w_arr[index_w] = t_w_arr[index_t];
		}
	}

	checkCudaErrors( cudaMemcpy(dv_w_arr, w_arr, bc_w_arr, cudaMemcpyHostToDevice) );
}

void AnnCUDA2::printf_Network(string filename){
  FILE * pFile;
  const char * c = filename.c_str();
  pFile = fopen(c, "wb");
  cTopology->printTopology(pFile);

  int weightCount = cTopology->obtainWeightCount();

  double *w_arr_dbl = new double[weightCount];
  double *dw_arr_dbl = new double[weightCount];
	for(int layer_id = 0; layer_id < L - 1; layer_id++)
		for(int k = 0; k < l_real[layer_id]*(l_real[layer_id+1]-1); k++){
			w_arr_dbl[sw_real[layer_id]+k] = (double)w_arr[sw[layer_id]+k];
			dw_arr_dbl[sw_real[layer_id]+k] = (double)dw_arr[sw[layer_id]+k];
  	}

  fwrite (w_arr_dbl , sizeof(double), weightCount, pFile);
  fwrite (dw_arr_dbl , sizeof(double), weightCount, pFile);
  fclose (pFile);
}
