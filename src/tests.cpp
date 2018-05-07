
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
  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialDBL  *serialDBL = new AnnSerialDBL(topology);

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

  serialDBL->setWeights(warr);

  serialDBL->printf_Network("w_and_dw_tests.bin");
  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];

	serialDBL->feedForward(input, output);


  //              0.73622649825740200000
  if(output[0] != 0.73622649825740249518) return false;
  //printf("%.10f\n", output[0]);

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

  AnnSerialDBL  *serialDBL = new AnnSerialDBL(topology);
  double alpha = 0.7;
  double eta = 0.25;

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

  serialDBL->setWeights(warr);

  double *input = new double[2];
  input[0] = 1;
  input[1] = 2;

  double *output = new double[1];
  output[0] = 0.5;


	serialDBL->train(input, output, alpha, eta);

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

  // for (int i = 0; i < 9; i++){
  //   printf("%.10f\n", warr2[i]);
  // }


  if(serialDBL->getA()[2] != 1) return false;
  if(serialDBL->getA()[5] != 1) return false;

  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialDBL;
  delete topology;
}

bool test_forwardFLT(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialFLT  *serialFLT = new AnnSerialFLT(topology);

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


  serialFLT->setWeights(warr);

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

  AnnSerialFLT  *serialFLT = new AnnSerialFLT(topology);
  float alpha = 0.7;
  float eta = 0.25;

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

  serialFLT->setWeights(warr);

  float *input = new float[2];
  input[0] = 1;
  input[1] = 2;

  float *output = new float[1];
  output[0] = 0.5;


	serialFLT->train(input, output, alpha, eta);


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

bool test_forwardCUDA(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnCUDA  *serialFLT = new AnnCUDA(topology);

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


  serialFLT->setWeights(warr);

  float *input = new float[2];
  input[0] = 1;
  input[1] = 2;

  float *output = new float[1];


	serialFLT->feedForward(input, output);

//  printf("output = %.20f\n", output[0]);

  //              0.7362264983
  //              0.73622649908065795898
  if(output[0] != 0.73622649908065795898) return false;


  delete [] warr;
  delete [] input;
  delete [] output;
  delete serialFLT;
  delete topology;
}

bool test_backwardCUDA(){

  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnCUDA  *serialFLT = new AnnCUDA(topology);
  float alpha = 0.7;
  float eta = 0.25;

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

  serialFLT->setWeights(warr);

  float *input = new float[2];
  input[0] = 1;
  input[1] = 2;

  float *output = new float[1];
  output[0] = 0.5;


	serialFLT->train(input, output, alpha, eta);


  float *warr2 = serialFLT->getWeights();
  float *w = new float[9];
  // for(int i = 0; i < 9; i++)
  //   printf("w[%d] = %.20f\n", i, warr2[i]);

     w[0] = 0.49771153926849365234;
     w[1] = 0.19935533404350280762;
     w[2] = -0.00457693310454487801;
     w[3] = 0.09871067851781845093;
     w[4] = 0.19771154224872589111;
     w[5] = 0.69935530424118041992;
     w[6] = 0.89233678579330444336;
     w[7] = 0.29139557480812072754;
     w[8] = 0.18853138387203216553;

     // for(int i = 0; i < 9; i++){
     //   printf("%d\n", i);
     //   printf("%.20f\n", w[i]);
     //   printf("%.20f\n", warr2[i]);
     // }

      //Cuda results
      // w[0] = 0.49847087264060974121
      // w[1] = 0.19956924021244049072
      // w[2] = -0.00171694520395249128
      // w[3] = 0.09951633960008621216
      // w[4] = 0.19771154224872589111
      // w[5] = 0.69935530424118041992
      // w[6] = 0.89999997615814208984
      // w[7] = 0.30000001192092895508
      // w[8] = 0.20000000298023223877

    //CUDA
    //             0.49771153302267300
    // if(warr2[0] != 0.49847087264060974121) return false;
    //             0.19935533771596700
    // if(warr2[1] != 0.19956924021244049072) return false;
    //             -0.00457693395465348
    // if(warr2[2] != -0.00171694520395249128) return false;
    //             0.09871067543193330
    // if(warr2[3] != 0.09951633960008621216) return false;
    //             0.19771153302267300
    // if(warr2[4] != 0.19771154224872589111) return false;
    //             0.69935533771596700
    // if(warr2[5] != 0.69935530424118041992) return false;
    //             0.89233680716791600
    //if(warr2[6] != 0.89999997615814208984) return false;
    //             0.29139555061784700
    // if(warr2[7] != 0.30000001192092895508) return false;
    //             0.18853137822738400
    // if(warr2[8] != 0.20000000298023223877) return false;


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
  Topology *topology = new Topology();
  topology->addLayer(2);
  topology->addLayer(2);
  topology->addLayer(1);

  AnnSerialDBL  *serialDBL = new AnnSerialDBL(topology);

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


  serialDBL->setWeights(warr);

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

  Topology* tp=serialDBL_2->getTopology();

  if(tp->getLayerSize(0)!=2) return false;
  if(tp->getLayerSize(1)!=2) return false;
  if(tp->getLayerSize(2)!=1) return false;

  double *warr2 = serialDBL_2->getWeights();
  for(int i=0;i<9;i++){
      // printf("%.10f    %.10f\n", warr[i],warr2[i]);
      if(warr[i]!=warr2[i]){
       return false;
     }
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
  delete tp;

  return true;
}

bool test_Topology_From_File(){
  AnnSerialDBL *serialDBL_2=new AnnSerialDBL("w_and_dw_tests.bin");

  Topology* tp=serialDBL_2->getTopology();
  if(tp->getLayerSize(0)!=2) return false;
  if(tp->getLayerSize(1)!=2) return false;
  if(tp->getLayerSize(2)!=1) return false;

  delete serialDBL_2;
  delete tp;
  return true;
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

  passed = test_forwardCUDA(); failCount += passed ? 0 : 1;
  printf("%s - test_forward_CUDA\n", passed ? "PASSED" : "FAILED");

  passed = test_backwardCUDA(); failCount += passed ? 0 : 1;
  printf("%s - test_backward_CUDA\n", passed ? "PASSED" : "FAILED");


  passed = test_Topology_From_File(); failCount += passed ? 0 : 1;
  printf("%s - test_Topology_From_File\n", passed ? "PASSED" : "FAILED");

  passed = test_File_Double(); failCount += passed ? 0 : 1;
  printf("%s - test_Files_Double\n", passed ? "PASSED" : "FAILED");

  printf("\n");
  if(failCount == 0) printf("ALL tests PASSED\n");
  else printf("%d TESTS FAILED\n", failCount);

  return failCount == 0;
}
