#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef __HANDLE_ERROR_H__
	#define __HANDLE_ERROR_H__

	#ifdef __cplusplus
	extern "C" {
		#include <cstddef>
	#endif

		/*-------------------------------------------------------------------
		* Function to process CUDA errors
		* @param err [IN] CUDA error to process (usually the code returned
		* by the cuda function)
		* @param line [IN] line of source code where function is called
		* @param file [IN] name of source file where function is called
		* @return on error, the function terminates the process with
		* EXIT_FAILURE code
		* source: "CUDA by Example: An Introduction to General-Purpose "
		* GPU Programming", Jason Sanders, Edward Kandrot, NVIDIA, July 2010"
		* @note: the function should be called through the
		* macro 'HANDLE_ERROR'
		*------------------------------------------------------------------*/
		static void HandleError( cudaError_t err, const char *file, int line ) {
			if (err != cudaSuccess)
			{
				printf( "[ERROR] '%s' (%d) in '%s' at line '%d'\n", cudaGetErrorString(err),err,file,line);
				exit( EXIT_FAILURE );
			}
		}
		/* The HANDLE_ERROR macro */
		#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))

#ifdef __cplusplus
	}
#endif
#endif /* __HANDLE_ERROR_H__ */
