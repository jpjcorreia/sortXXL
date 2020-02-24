/**
 * @file module_bitonic_sort.cu
 * @brief source code file for the bitonic sort algorithm implementation
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../3rd/sortXXL_cmd.h"
#include "../3rd/HandleError.h"
#include "../3rd/debug.h"

#include "constants.h"
#include "aux.h"
#include "module_bitonic_sort.h"

// n limited to 2^22=4194304 for 128 threads -> 4194304/128=32768
float bitonic_sort(int **values, int n){
	size_t size = n * sizeof(int);
	int *dev_values, *resized_values, k, j, padded_size=n, hardware_sorting_capabilities=0, max_treads_to_use=0;
	cudaEvent_t start, stop;
    float elapsedTime;
    cudaDeviceProp prop;
	int deviceToUse=-1;

	resized_values = *values;

	// Calculate the padding to align the number of data elements to the next power of two of row count
	padded_size = get_next_power_of_two(n);

	// Let's check the hardware capabilities
	prop.regsPerBlock = padded_size; // we want the device with closest number of registers of number rows to sort
	prop.maxThreadsPerBlock = padded_size; // we want the device with closest number of threads of number rows to sort
	HANDLE_ERROR (cudaChooseDevice(&deviceToUse, &prop));
	HANDLE_ERROR (cudaGetDeviceProperties(&prop,deviceToUse));

	// Let's test if can sort this data with half of the threads
	max_treads_to_use = prop.maxThreadsPerBlock/2;

	hardware_sorting_capabilities = prop.regsPerBlock*max_treads_to_use;
	if(hardware_sorting_capabilities<padded_size){
		DEBUG("\nWe can't sort this data with half the threads... Let's try with all of them.");

		// Let's test with all the threads (possible more slow, but maybe it will get the work done)
		max_treads_to_use = prop.maxThreadsPerBlock;
		hardware_sorting_capabilities = prop.regsPerBlock*max_treads_to_use;
		if(hardware_sorting_capabilities<padded_size){
			fprintf(stderr, "\nThe GPU doesn't have the necessary compute capabilities to sort this data. \nThis software in this hardware is capable to sort a maximum of %d numbers.\n", hardware_sorting_capabilities);
			fflush(stderr);
			exit(UNSUFFICENT_GPU_CAMPABILITIES);
		}
	}



	// Add padding to align the number of data elements to a power of two
	if(padded_size>n){
		// Allocate more memory for the padding
		resized_values = (int*) realloc(*values, padded_size*sizeof(int));
		if (resized_values!=NULL) {

			// Align the data
			pad_data_to_align(resized_values, n, padded_size);

			// Update the size
			size = padded_size*sizeof(int);
		}else{
			MY_DEBUG("Problem while allocating the necessary memory for padding (memory exhausted?)\n");
		}
	}

	// allocate device memory
	HANDLE_ERROR( cudaMalloc((void**)&dev_values, size) );

	// copy data to device
	HANDLE_ERROR( cudaMemcpy(dev_values, resized_values, size, cudaMemcpyHostToDevice) );

	MY_DEBUG("Beginning kernel execution...\n");

	// Create the timers
	HANDLE_ERROR (cudaEventCreate (&start));
	HANDLE_ERROR (cudaEventCreate (&stop));

	// Synchronize the threads
	HANDLE_ERROR( cudaThreadSynchronize() );

	// Start the timer
	HANDLE_ERROR (cudaEventRecord (start, 0));

	// execute kernel
	for (k = 2; k <= padded_size; k <<= 1) {
		for (j = k >> 1; j > 0; j = j >> 1) {
			if (padded_size < max_treads_to_use)
				cuda_bitonic_sort <<< 1, padded_size >>> (dev_values, j, k, padded_size);
			else
				cuda_bitonic_sort <<< padded_size / max_treads_to_use, max_treads_to_use >>> (dev_values, j, k, padded_size);
		}
	}

	// Wait for synchronization of all threads
	HANDLE_ERROR( cudaThreadSynchronize() );

	/* Terminate the timer */
	HANDLE_ERROR (cudaEventRecord (stop, 0));
	HANDLE_ERROR (cudaEventSynchronize (stop));
	HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	HANDLE_ERROR (cudaEventDestroy (start));
	HANDLE_ERROR (cudaEventDestroy (stop));

	// Copy data back to host
	HANDLE_ERROR( cudaMemcpy(resized_values, dev_values, size, cudaMemcpyDeviceToHost) );

	// Update with the sorted values
	*values = resized_values;

	// Remove the padding and free some memory
	if(padded_size>n){
		padded_size=n;
		// Allocate more memory for the padding
		resized_values = (int*) realloc(*values, padded_size*sizeof(int));
		if (resized_values!=NULL) {
			// Update the size
			size = padded_size*sizeof(int);

			// Update the reference
			*values = resized_values;
		}
	}

	// free memory
	HANDLE_ERROR( cudaFree(dev_values) );

	// Free device resources (replaces depreciated cudaThreadExit)
	cudaDeviceReset();

	// Return the elapsed time
	return elapsedTime;
}

// Kernel function
__global__ void cuda_bitonic_sort(int* values, int j, int k, int n) {
	const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		int ixj = idx^j;
 		if (ixj > idx) {
			if ((idx&k) == 0 && values[idx] > values[ixj]) {
				//exchange(idx, ixj);
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
			if ((idx&k) != 0 && values[idx] < values[ixj]) {
				//exchange(idx, ixj);
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
		}
	}
}
