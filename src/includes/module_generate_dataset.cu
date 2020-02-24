/**
 * @file module_generate_dataset.cu
 * @brief source code file for generating a data set of random numbers
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include "../3rd/HandleError.h"
#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "module_generate_dataset.h"

/**
 * @brief provides an error management facilitator for CURAND calls
 */
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int generate_dataset_from_arguments(struct gengetopt_args_info args_info, int *data){
	int minimum = 0, maximum = INT_MAX;

	if (args_info.min_given == 1){
		minimum = args_info.min_arg;
	}

	if (args_info.max_given == 1){
		maximum = args_info.max_arg;
	}

	if(args_info.random_given == 1 && args_info.random_arg>0){
		return generate_dataset_in_interval(data, args_info.random_arg, minimum, maximum);
	}
	return EXIT_FAILURE;
}

int generate_dataset(int *data, int n){
	return generate_dataset_in_interval(data, n, 0, INT_MAX);
}

// based on http://docs.nvidia.com/cuda/curand/index.html#topic_1_3_1 and http://aresio.blogspot.pt/2011/05/cuda-random-numbers-inside-kernels.html
int generate_dataset_in_interval(int *data, int n, int minimum, int maximum){
	int i;
	curandGenerator_t gen;
	float *devData, *hostData;

	/* Allocate n floats on host */
	hostData = (float *)calloc(n, sizeof(float));

	/* Allocate n floats on device */
	HANDLE_ERROR(cudaMalloc((void **)&devData, n*sizeof(float)));

	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	/* Set seed */// added the time as seed
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	/* Generate n floats on device */
	CURAND_CALL(curandGenerateUniform(gen, devData, n));

	/* Copy device memory to host */
	HANDLE_ERROR(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));

	/* Copy the result */// enforced the limits
	for(i = 0; i < n; i++) {
		data[i] = (int) round((minimum + hostData[i] * (maximum-minimum)));
	}

	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gen));
	HANDLE_ERROR(cudaFree(devData));
	free(hostData);
	return EXIT_SUCCESS;
}
