
#include <stdio.h>
#include "tests.h"

bool test_topology(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  if(topology->obtainNeuronCount() != 8) return false;
  if(topology->obtainWeightCount() != 9) return false;


  return true;
}

bool test_forward_Double(){

  // int arr[] = {2,2,1};
	// setTopology(topolygy, arr, sizeof(arr)/sizeof(arr[0]));
  // return true;

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialDBL  *serialDBL = new AnnSerialDBL();

  double alpha = 0.7;
  double eta = 0.25;
  serialDBL->prepare(topology, alpha, eta);

  double *warr = new double[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;


  serialDBL->init(warr);

  serialDBL->printf_Network("");
  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];


	serialDBL->feedForward(input, output);

  //printf("output = %.20f\n", output[0]);

  //              0.73622649825740200000
  if(output[0] != 0.73622649825740249518) return false;

  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;

  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology;
}

bool test_backward_Double(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialDBL  *serialDBL = new AnnSerialDBL();
  double alpha = 0.7;
  double eta = 0.25;
  serialDBL->prepare(topology, alpha, eta);

  double *warr = new double[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;


  serialDBL->init(warr);

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];
  output[0] = 0.5;


	serialDBL->train(input, output);

  //printf("output = %.20f\n", output[0]);


  double *warr2 = serialDBL->getWeights();
  // for(int i = 0; i < 9; i++)
  //  printf("w[%d] = %.20f\n", i, warr2[i]);

  // w[0] = 0.49771153302267323593
  // w[1] = 0.19935533771596666841
  // w[2] = -0.00457693395465348218
  // w[3] = 0.09871067543193333405
  // w[4] = 0.19771153302267327478
  // w[5] = 0.69935533771596658514
  // w[6] = 0.89233680716791641263
  // w[7] = 0.29139555061784666590
  // w[8] = 0.18853137822738402773



  //             0.49771153302267300
  if(warr2[0] != 0.49771153302267323593) return false;
  //             0.19935533771596700
  if(warr2[1] != 0.19935533771596666841) return false;
  //             -0.00457693395465348
  if(warr2[2] != -0.00457693395465348218) return false;
  //             0.09871067543193330
  if(warr2[3] != 0.09871067543193333405) return false;
  //             0.19771153302267300
  if(warr2[4] != 0.19771153302267327478) return false;
  //             0.69935533771596700
  if(warr2[5] != 0.69935533771596658514) return false;
  //             0.89233680716791600
  if(warr2[6] != 0.89233680716791641263) return false;
  //             0.29139555061784700
  if(warr2[7] != 0.29139555061784666590) return false;
  //             0.18853137822738400
  if(warr2[8] != 0.18853137822738402773) return false;


  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;

  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology;
}

bool test_forwardFLT(){

  // int arr[] = {2,2,1};

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialFLT  *serialFLT = new AnnSerialFLT();

  double alpha = 0.7;
  double eta = 0.25;
  serialFLT->prepare(topology, alpha, eta);

  float *warr = new float[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;


  serialFLT->init(warr);

  float *input = new float[2];
  input[0] = 1;
  input[1] = 2;

  float *output = new float[1];


	serialFLT->feedForward(input, output);

  //printf("output = %.20f\n", output[0]);

  //              0.7362264983
  if(output[0] != 0.73622649908065795898) return false;



  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialFLT;
  delete topology;
}

bool test_backwardFLT(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialFLT  *serialFLT = new AnnSerialFLT();
  float alpha = 0.7;
  float eta = 0.25;
  serialFLT->prepare(topology, alpha, eta);

  float *warr = new float[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;


  serialFLT->init(warr);

  float *input = new float[2];
  input[0] = 1;
  input[1] = 2;

  float *output = new float[1];
  output[0] = 0.5;


	serialFLT->train(input, output);

  //printf("output = %.20f\n", output[0]);


  float *warr2 = serialFLT->getWeights();
  //for(int i = 0; i < 9; i++)
  // printf("w[%d] = %.20f\n", i, warr2[i]);

   // w[0] = 0.49771153926849365234
   // w[1] = 0.19935533404350280762
   // w[2] = -0.00457693310454487801
   // w[3] = 0.09871067851781845093
   // w[4] = 0.19771154224872589111
   // w[5] = 0.69935530424118041992
   // w[6] = 0.89233678579330444336
   // w[7] = 0.29139557480812072754
   // w[8] = 0.18853138387203216553


  //             0.49771153302267300
  if(warr2[0] != 0.49771153926849365234) return false;
  //             0.19935533771596700
  if(warr2[1] != 0.19935533404350280762) return false;
  //             -0.00457693395465348
  if(warr2[2] != -0.00457693310454487801) return false;
  //             0.09871067543193330
  if(warr2[3] != 0.09871067851781845093) return false;
  //             0.19771153302267300
  if(warr2[4] != 0.19771154224872589111) return false;
  //             0.69935533771596700
  if(warr2[5] != 0.69935530424118041992) return false;
  //             0.89233680716791600
  if(warr2[6] != 0.89233678579330444336) return false;
  //             0.29139555061784700
  if(warr2[7] != 0.29139557480812072754) return false;
  //             0.18853137822738400
  if(warr2[8] != 0.18853138387203216553) return false;

  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialFLT;
  delete topology;
}

bool test_File_Double(){
  // int arr[] = {2,2,1};
	// setTopology(topolygy, arr, sizeof(arr)/sizeof(arr[0]));
  // return true;

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialDBL  *serialDBL = new AnnSerialDBL();

  double alpha = 0.7;
  double eta = 0.25;
  serialDBL->prepare(topology, alpha, eta);

  double *warr = new double[9];
  int idx = 0;
  warr[idx++] = 0.5;
  warr[idx++] = 0.2;
  warr[idx++] = 0.0;

  warr[idx++] = 0.1;
  warr[idx++] = 0.2;
  warr[idx++] = 0.7;

  warr[idx++] = 0.9;
  warr[idx++] = 0.3;
  warr[idx++] = 0.2;


  serialDBL->init(warr);

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];


	serialDBL->feedForward(input, output);

  //              0.73622649825740200000
  if(output[0] != 0.73622649825740249518) return false;

  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;


  //*********************************************************************************
  AnnSerialDBL *serialDBL_2=new AnnSerialDBL("w_and_dw_tests.bin");

  serialDBL_2->prepare(topology, alpha, eta);

  serialDBL_2->init(NULL);

  double *warr2 = serialDBL_2->getWeights();
  for(int i=0;i<9;i++){
    //  printf("w[%d] = %.20f\n", i, warr2[i]);
      if(warr[i]!=warr2[i])
       return false;
  }

  input[0] = 1;
  input[1] = 2;

  serialDBL_2->feedForward(input, output);

  if(output[0] != 0.73622649825740249518) return false;

  if(serialDBL_2->getA()[2] != 1) return false;
  if(serialDBL_2->getA()[5] != 1) return false;

  delete [] warr;
  delete [] warr2;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete serialDBL_2;
  delete topology;
}

bool run_tests(){

  printf("running tests ... \n");

  int failCount = 0;

  bool passed = test_topology(); failCount += passed ? 0 : 1;
  printf("%s - test_topology\n", passed ? "PASSED" : "FAILED");

  passed = test_forward_Double(); failCount += passed ? 0 : 1;
  printf("%s - test_forward_Double\n", passed ? "PASSED" : "FAILED");

  passed = test_backward_Double(); failCount += passed ? 0 : 1;
  printf("%s - test_backward_Double\n", passed ? "PASSED" : "FAILED");

  passed = test_forwardFLT(); failCount += passed ? 0 : 1;
  printf("%s - test_forward_Float\n", passed ? "PASSED" : "FAILED");

  passed = test_backwardFLT(); failCount += passed ? 0 : 1;
  printf("%s - test_backward_Float\n", passed ? "PASSED" : "FAILED");

  passed = test_File_Double(); failCount += passed ? 0 : 1;
  printf("%s - test_Files_Double\n", passed ? "PASSED" : "FAILED");

  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
