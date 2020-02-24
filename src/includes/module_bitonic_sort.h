/**
 * @file module_bitonic_sort.h
 * @brief header file for the bitonic sort algorithm
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_BITONIC_SORT_H
	#define	MODULE_BITONIC_SORT_H
	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif

			/**
			 * @brief Sort an array of integers using the bitonic algorithm
			 * @param values with the pointer for the address to starting ordering the data
			 * @param n with the number of elements in the array
			 * @return float with the elapsed time in the sort process
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			float bitonic_sort(int **values, int n);

	#ifdef __cplusplus
		}
	#endif

		/**
		 * @brief Generates a random data set of integers in a specific interval
		 * @param values with the pointer for the address were to store the data
		 * @param j
		 * @param k
		 * @param n with the number of elements in the array
		 * @author Cláudio Esperança <cesperanc@gmail.com>
		 */
		__global__ static void cuda_bitonic_sort(int* values, int j, int k, int n);

#endif	/* MODULE_BITONIC_SORT_H */

