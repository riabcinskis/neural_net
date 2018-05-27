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

	// CUDA2
	__global__ void
	kernel_feedforward_2(
		int layer_id,
		int *l,
		int *s_ext,
		int *sw_ext,
		float *z_ext_arr,
		float *a_ext_arr,
		float *w_ext_arr
	 ){

		 extern __shared__ int sm[];
		 float *sm_z = (float*)&sm[0];

		 int h = blockDim.x;
		 int h2 = blockDim.y;


		 int lidx = threadIdx.x;
		 int pidx = threadIdx.y;
		int idx = threadIdx.x + blockDim.x*blockIdx.x;

		int neuron_count = l[layer_id];
		int neuron_count_prev = l[layer_id-1];

		//printf("layer = %d idx = %d count = %d\n", layer_id, idx, neuron_count-1);
		if(idx >= neuron_count-1) return;

		float z = 0;
		int index0=sw_ext[layer_id-1];
		int index1=s_ext[layer_id-1];
		for(int k = pidx; k < neuron_count_prev; k+=h2){
			z += w_ext_arr[index0 + k*(neuron_count - 1) + idx]*a_ext_arr[index1 + k];
		}

		sm_z[pidx*h + lidx] = z;


		__syncthreads();

		if(pidx == 0){
			z = 0;
			for(int i = 0; i < h2; i++)
				z += sm_z[i*h + lidx];

				z_ext_arr[s_ext[layer_id] + idx] = z;
				float a = 1.0 / (1.0 + expf(-z));
				a_ext_arr[s_ext[layer_id] + idx] = a;
		}


		// printf("index = %d z = %.5f\n", s[layer_id] + idx, z);
		// printf("a = %.20f\n", a);
	}

	__global__ void
	kernel_calc_gL_2(
		int layer_id,
		int *l,
		int *s_ext,
		float *z_ext_arr,
		float *a_ext_arr,
		float *t_arr,
		float *gjl_ext
	 ){

		volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

		int neuron_count = l[layer_id];

		if(idx >= neuron_count-1) return;

		float z = z_ext_arr[s_ext[layer_id] + idx];
		float tmp = 1 + expf(-z);
		float f_deriv=expf(-z) / (tmp*tmp);

		gjl_ext[s_ext[layer_id] + idx] = f_deriv*(a_ext_arr[s_ext[layer_id] + idx] - t_arr[idx]);
	}

	__global__ void
	kernel_calc_gjL_2(
		int layer_id,
		int *l,
		int *s_ext,
		int *sw_ext,
		float *z_ext_arr,
		float *a_ext_arr,
		float *t_arr,
		float *gjl_ext,
		float *w_ext_arr
	 ){

		 int idx = threadIdx.y + blockDim.y*blockIdx.y;
 		int h = blockDim.x;
 		int pidx = threadIdx.y;
		int lidx = threadIdx.x;

		extern __shared__ int sm[];
		float *sm_g = (float*)&sm[0];


		int neuron_count = l[layer_id];
		int neuron_count_next = l[layer_id+1];

		if(idx >= neuron_count-1) return;

		float sum = 0;
		for (int k = lidx; k < neuron_count_next-1; k+=h) {
				sum += w_ext_arr[sw_ext[layer_id] + idx*(l[layer_id + 1] - 1) + k] * gjl_ext[s_ext[layer_id + 1] + k];
		}

		sm_g[pidx*h + lidx] = sum;

		__syncthreads();

		if(lidx == 0){
			float z = z_ext_arr[s_ext[layer_id] + idx];
			float tmp = 1 + expf(-z);
			float f_deriv = expf(-z) / (tmp*tmp);

			sum = 0;
			for(int i = 0; i < h; i++)
				sum += sm_g[pidx*h + i];


			gjl_ext[s_ext[layer_id] + idx] = f_deriv*sum;
		}
	}


	__global__ void
	kernel_weight_update_2(
		int layer_id,
		int *l,
		int *s_ext,
		int *sw_ext,
		float *z_ext_arr,
		float *a_ext_arr,
		float *t_arr,
		float *gjl_ext,
		float *w_ext_arr,
		float *dw_ext_arr,
		float eta,
		float alpha
	 ){

		 int idx = threadIdx.y + blockDim.y*blockIdx.y;
		 int h = blockDim.x;
		 int pidx=threadIdx.x;


		 int neuron_count = l[layer_id];
		 int neuron_count_next = l[layer_id+1];

		 if(idx >= neuron_count) return;

		 float a = a_ext_arr[s_ext[layer_id] + idx];

		 int index0 = s_ext[layer_id + 1] + pidx;
		 int index1 = sw_ext[layer_id] + idx*(neuron_count_next - 1) + pidx;
		 for(int k = pidx; k < neuron_count_next-1; k+=h){

			 float grad = a*gjl_ext[index0];
			 index0 += h;
			 float dw = dw_ext_arr[index1] = -eta*grad + alpha*dw_ext_arr[index1];

			 w_ext_arr[index1] += dw;


				index1 += h;

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
	h = 32;
	h2 = 32;

	cTopology = top;


	l = new int[top->getLayerCount()];
	l_ext = new int[top->getLayerCount()];
	s_ext = new int[top->getLayerCount()];


	int neuronCount = cTopology->obtainNeuronCount();
	int weightCount = cTopology->obtainWeightCount();

	int neuronCount_ext = obtainNeuronCountExt(cTopology);
	int weightCount_ext = obtainWeightCountExt(cTopology);

	// printf("neuronCount = %d\n", neuronCount);
	// printf("neuronCount2 = %d\n", neuronCount2);
	// printf("weightCount = %d\n", weightCount);
	// printf("weightCount2 = %d\n", weightCount2);

	a_ext_arr = new float[neuronCount_ext];
	z_ext_arr = new float[neuronCount_ext];

	sw_ext = new int[top->getLayerCount()];


	w_ext_arr = new float[weightCount_ext];
	dw_ext_arr = new float[weightCount_ext];

	t_arr = new float[top->getLayerSize(top->getLayerCount() - 1)];

	gjl_ext = new float[neuronCount_ext];

	// cuda

	int deviceCount = 0;
	checkCudaErrors( cudaGetDeviceCount(&deviceCount));
	if(deviceCount == 0){
		printf("*** there is no CUDE device\n");
		return;
	}

	checkCudaErrors( cudaSetDevice(0) );

	dv_l = NULL; bc_l = sizeof(int)*top->getLayerCount();
	dv_s_ext = NULL; bc_s_ext = sizeof(int)*top->getLayerCount();;

	dv_a_ext_arr = NULL; bc_a_ext_arr = sizeof(float)*neuronCount_ext;
	dv_z_ext_arr = NULL; bc_z_ext_arr = sizeof(float)*neuronCount_ext;

	dv_sw_ext = NULL; bc_sw_ext = sizeof(int)*top->getLayerCount();

	dv_w_ext_arr = NULL; bc_w_ext_arr = sizeof(float)*weightCount_ext;
	dv_dw_ext_arr = NULL; bc_dw_ext_arr = sizeof(float)*weightCount_ext;

	dv_t_arr = NULL; bc_t_arr = sizeof(float)*top->getLayerSize(top->getLayerCount() - 1);
	dv_gjl_ext = NULL; bc_gjl_ext = sizeof(float)*neuronCount_ext;

	checkCudaErrors( cudaMalloc((void **)&dv_l, bc_l) );
	checkCudaErrors( cudaMalloc((void **)&dv_s_ext, bc_s_ext) );
	checkCudaErrors( cudaMalloc((void **)&dv_a_ext_arr, bc_a_ext_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_z_ext_arr, bc_z_ext_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_sw_ext, bc_sw_ext) );
	checkCudaErrors( cudaMalloc((void **)&dv_w_ext_arr, bc_w_ext_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_dw_ext_arr, bc_dw_ext_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_t_arr, bc_t_arr) );
	checkCudaErrors( cudaMalloc((void **)&dv_gjl_ext, bc_gjl_ext) );

}

void AnnCUDA2::init(FILE *pFile=NULL){
  L = cTopology->getLayerCount();

	int *W = new int[L];
	int *W_ext = new int[L];


	Random *rnd = new Random();

	//Neuronu kiekiai sluoksnyje
	for (int i = 0; i < L; i++) {
		int neuron_count = cTopology -> getLayerSize(i) + 1;
		l[i] = neuron_count;
		l_ext[i] = neuron_count + (32 - neuron_count % 32);
	}

	//Sluoksniu pradzios indeksai
	for (int i = 0; i < L; i++) {
		s_ext[i] = 0;
		for (int j = i; j > 0; j--) {
			s_ext[i] += l_ext[j - 1];
		}
	}

	//Bias neuronai
	for (int i = 0; i < L - 1; i++) {
		a_ext_arr[s_ext[i] + l[i] - 1] = 1;
	}

	//Svoriu kiekiai l-ame sluoksnyje
	for (int i = 0; i < L - 1; i++) {

		W[i] = l[i] * (l[i + 1] - 1);
		W_ext[i] = 	W[i];
		if (W_ext[i] % 32 != 0) {
			W_ext[i] += (32 - W_ext[i] % 32);
		}
		sw_ext[i] = 0;
		if (i != 0) {
			for (int j = 0; j < i; j++) {
				sw_ext[i] += W_ext[j];

			}
		}
  }

  for (int i = 0; i < L - 1; i++)
    for (int j = 0; j < W_ext[i]; j++) {
			if (j < W[i]){
      	w_ext_arr[sw_ext[i] + j] =(rnd->next()*2-1);
			}
			else{
				w_ext_arr[sw_ext[i] + j] = 0.0;
			}
      dw_ext_arr[sw_ext[i] + j] = 0.0;
  }

	delete [] W;
	delete [] W_ext;

	checkCudaErrors( cudaMemcpy(dv_w_ext_arr, w_ext_arr, bc_w_ext_arr, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_dw_ext_arr, dw_ext_arr, bc_dw_ext_arr, cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemcpy(dv_l, l, bc_l, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_s_ext, s_ext, bc_s_ext, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(dv_sw_ext, sw_ext, bc_sw_ext, cudaMemcpyHostToDevice) );
}

void AnnCUDA2::train(float *a, float *b, float alpha, float eta){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_ext_arr[i] = a[i];
	}

	for (int j = 0; j < obtainNeuronCountExt(cTopology); j++) {
		z_ext_arr[j] = 0;
	}

	calc_feedForward();

	for (int i = 0; i < cTopology->getLayerSize(cTopology->getLayerCount() - 1); i++) {
		t_arr[i] = b[i];
	}

	calc_gjl();

	for (int i = 0; i < L-1; i++) {//per sluoksnius einu+

		int neuron_count = l[i];
		int g = (neuron_count + (h2-neuron_count%h2))/h2; // number of grids
		dim3 grid_dim(1, g, 1);
		dim3 block_dim(h, h2, 1);
		ann::kernel_weight_update_2<<<grid_dim, block_dim>>>(
			i,
			dv_l,
			dv_s_ext,
			dv_sw_ext,
			dv_z_ext_arr,
			dv_a_ext_arr,
			dv_t_arr,
			dv_gjl_ext,
			dv_w_ext_arr,
			dv_dw_ext_arr,
			eta,
			alpha
		);
	//	checkCudaErrors( cudaDeviceSynchronize() );
	}

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

}

void AnnCUDA2::finishTraining(){
	checkCudaErrors( cudaMemcpy(w_ext_arr, dv_w_ext_arr, bc_w_ext_arr, cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(dw_ext_arr, dv_dw_ext_arr, bc_dw_ext_arr, cudaMemcpyDeviceToHost) );
}

void AnnCUDA2::feedForward(float *a, float *b){
	for (int i = 0; i < cTopology->getLayerSize(0); i++) {
		a_ext_arr[i] = a[i];
	}

	for (int j = 0; j < obtainNeuronCountExt(cTopology); j++) {
		z_ext_arr[j] = 0;
	}

	calc_feedForward();

	checkCudaErrors( cudaMemcpy(a_ext_arr, dv_a_ext_arr, bc_a_ext_arr, cudaMemcpyDeviceToHost) );

	for (int i = 0; i < l[L - 1]; i++){
		b[i] = a_ext_arr[s_ext[L - 1] + i];
		//printf("b[%d] = %.10f\n", i, b[i]);
	}
}

void AnnCUDA2::calc_feedForward(){

	checkCudaErrors( cudaMemcpy(dv_a_ext_arr, a_ext_arr, bc_a_ext_arr, cudaMemcpyHostToDevice) );

	for (int i = 1; i < L; i++) {//per sluoksnius einu+

		int neuron_count = l[i];

	  int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
		dim3 grid_dim(g, 1, 1);
		dim3 block_dim(h, h2, 1);
		int bc_sm = sizeof(float)*h*h2;
		ann::kernel_feedforward_2<<<grid_dim, block_dim, bc_sm>>>(
			i,
			dv_l,
			dv_s_ext,
			dv_sw_ext,
			dv_z_ext_arr,
			dv_a_ext_arr,
			dv_w_ext_arr
		);
	//	checkCudaErrors( cudaDeviceSynchronize() );
	}
}

void AnnCUDA2::calc_gjl(){

	checkCudaErrors( cudaMemcpy(dv_t_arr, t_arr, bc_t_arr, cudaMemcpyHostToDevice) );

	int last_layer_id = L-1;
	int neuron_count = l[last_layer_id];
	int g = (neuron_count + (h-neuron_count%h))/h; // number of grids
	dim3 grid_dim(g, 1, 1);
	dim3 block_dim(h, 1, 1);
	ann::kernel_calc_gL_2<<<grid_dim, block_dim>>>(
		last_layer_id,
		dv_l,
		dv_s_ext,
		dv_z_ext_arr,
		dv_a_ext_arr,
		dv_t_arr,
		dv_gjl_ext
	);
	//checkCudaErrors( cudaDeviceSynchronize() );

	for (int i = L - 2; i >= 1; i--) {
			neuron_count = l[i];
			g = (neuron_count + (h2-neuron_count%h2))/h2; // number of grids
			dim3 grid_dim(1, g, 1);
			dim3 block_dim(h, h2, 1);
			int bc_sm = sizeof(float)*h*h2;
			ann::kernel_calc_gjL_2<<<grid_dim, block_dim,bc_sm>>>(
				i,
				dv_l,
				dv_s_ext,
				dv_sw_ext,
				dv_z_ext_arr,
				dv_a_ext_arr,
				dv_t_arr,
				dv_gjl_ext,
				dv_w_ext_arr
			);
		}
}

float AnnCUDA2::obtainError(float *b){
	checkCudaErrors( cudaMemcpy(a_ext_arr, dv_a_ext_arr, bc_a_ext_arr, cudaMemcpyDeviceToHost) );
	float error = 0;
	for(int i = 0; i < l[L-1] - 1; i++){
		float tmp = b[i] - a_ext_arr[s_ext[L-1] + i];
		error += tmp*tmp;
		//printf("a_arr[%d] = %.10f\n", s[L-1] + i, a_arr[s[L-1] + i]);
	}
	return error;
}

void AnnCUDA2::destroy(){
	delete[] l;
	l = NULL;

	delete[] l_ext;
	l_ext = NULL;


	delete[] s_ext;
	s_ext = NULL;

	delete[] a_ext_arr;
	a_ext_arr = NULL;
	delete[] z_ext_arr;
	z_ext_arr = NULL;

	delete[] sw_ext;
	sw_ext = NULL;

	delete[] w_ext_arr;
	w_ext_arr = NULL;
	delete[] dw_ext_arr;
	dw_ext_arr = NULL;

	delete[] t_arr;
	t_arr = NULL;

	delete[] gjl_ext;
	gjl_ext = NULL;



	checkCudaErrors( cudaFree(dv_l) );
	checkCudaErrors( cudaFree(dv_s_ext) );
	checkCudaErrors( cudaFree(dv_a_ext_arr) );
	checkCudaErrors( cudaFree(dv_z_ext_arr) );
	checkCudaErrors( cudaFree(dv_sw_ext) );
	checkCudaErrors( cudaFree(dv_w_ext_arr) );
	checkCudaErrors( cudaFree(dv_dw_ext_arr) );
	checkCudaErrors( cudaFree(dv_t_arr) );
	checkCudaErrors( cudaFree(dv_gjl_ext) );


  checkCudaErrors(cudaDeviceReset());
}

float* AnnCUDA2::getWeights(){
	return w_ext_arr;
}

float* AnnCUDA2::getA(){
	return a_ext_arr;
}

void AnnCUDA2::print_out(){
  printf("z = %e\n", z_ext_arr[s_ext[L-1]+0]);
	printf("g = %e\n", gjl_ext[s_ext[L-1]+0]);

	for(int i = 0; i < l[L-2]; i++){
		if(i < l[L-2]) printf("[%d] z=%e, a=%e, w=%e, grad = %e\n",
		 	i, z_ext_arr[s_ext[L-2]+i],
			a_ext_arr[s_ext[L-2]+i],
		  w_ext_arr[sw_ext[L-2] + i*(l[L-1]-1)],
			a_ext_arr[s_ext[L-2]+i]*gjl_ext[s_ext[L-1]+0]);
	}
}

void AnnCUDA2::setWeights(float *t_w_arr) {
	int prev_count = 0;
	for (int i = 0; i < cTopology->getLayerCount() - 1; i++) {

		for (int j = 0; j < l[i]*(l[i+1]-1); j++) {
			int index_w = sw_ext[i] + j;
			int index_t = prev_count + j;
			w_ext_arr[index_w] = t_w_arr[index_t];
		}
		prev_count += l[i]*(l[i+1]-1);

	}

	checkCudaErrors( cudaMemcpy(dv_w_ext_arr, w_ext_arr, bc_w_ext_arr, cudaMemcpyHostToDevice) );
}

void AnnCUDA2::printf_Network(string filename){
  FILE * pFile;
  const char * c = filename.c_str();
  pFile = fopen(c, "wb");
  cTopology->printTopology(pFile);

  int weightCount = cTopology->obtainWeightCount();

  double *w_arr_dbl = new double[weightCount];
  double *dw_arr_dbl = new double[weightCount];
	int sw_index = 0;
	for(int layer_id = 0; layer_id < L - 1; layer_id++){

		for(int k = 0; k < l[layer_id]*(l[layer_id+1]-1); k++){
			w_arr_dbl[sw_index+k] = (double)w_ext_arr[sw_ext[layer_id]+k];
			dw_arr_dbl[sw_index+k] = (double)dw_ext_arr[sw_ext[layer_id]+k];

		}
		sw_index +=  l[layer_id]*(l[layer_id+1]-1);
	}

  fwrite (w_arr_dbl , sizeof(double), weightCount, pFile);
  fwrite (dw_arr_dbl , sizeof(double), weightCount, pFile);
  fclose (pFile);
}

/* static */
int AnnCUDA2::obtainNeuronCountExt(Topology *top){
  int count = 0;
  for (int i = 0; i < top->getLayerCount(); i++){
    int neuron_count = top->getLayerSize(i)+1;
    count += neuron_count;
    if (neuron_count % 32 != 0)
      count += 32 - neuron_count % 32;
  }
  return count;
}

/* static */
int AnnCUDA2::obtainWeightCountExt(Topology *top){
  int count = 0;
  for (int i = 0; i < top->getLayerCount()-1; i++){
    int weight_count =  (top->getLayerSize(i)+1)*top->getLayerSize(i+1); //((*ml)[i] + 1)*(*ml)[i+1];
    count += weight_count;
    if (weight_count % 32 != 0)
      count += 32 - weight_count % 32;
  }
  return count;
}
