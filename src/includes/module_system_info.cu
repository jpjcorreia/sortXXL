/**
 * @file module_system_info.cu
 * @brief source code file for the system information module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

#include "../3rd/HandleError.h"
#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "module_system_info.h"

void system_info(struct gengetopt_args_info args_info){
	cudaDeviceProp prop;
	int count;

	if (args_info.gpu_given == 1){

		/* Get info from device */
		cudaGetDeviceCount(&count);
		for (int i=0; i< count; i++){
			HANDLE_ERROR (cudaGetDeviceProperties(&prop,i));
			printf( " --- Information for device %d ---\n", i );
			printf( "Name: %s\n", prop.name );
			printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
			printf( "Number of cores: %s\n", number_of_cores(prop.major, prop.minor) );
		    printf( "Clock rate: %d\n", prop.clockRate );
			printf( " --- Memory Information for device %d ---\n", i );
			printf( "Total global mem: %zu\n",prop.totalGlobalMem );
			printf( " --- MP Information for device %d ---\n", i );
			printf( "Multiprocessor count: %d\n",prop.multiProcessorCount );
		    printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		    printf( "Registers per mp: %d\n", prop.regsPerBlock );
		    printf( "Threads in warp: %d\n", prop.warpSize );
		    printf( "Max threads per block: %d\n", prop.maxThreadsPerBlock );
		    printf( "Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		    printf( "Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
		    printf( "\n" );
		}
	}
}

char* number_of_cores(double gpu_major, double gpu_minor){
	double cap = gpu_major*10+gpu_minor;
	if(cap <= 13){
		return "8";
	}else if(cap == 20){
		return "32";
	}else if(cap == 21){
		return "48";
	}else if(cap == 30){
		return "192";
	}else{
		return "unsupported";
	}
}
