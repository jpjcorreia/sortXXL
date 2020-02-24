/**
 * @file module_generate_dataset.h
 * @brief header file for the data set generation module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_GENERATE_DATASET_H
	#define	MODULE_GENERATE_DATASET_H

	#ifdef __cplusplus
		extern "C" {
	#include <cstddef>
	#endif
			/**
			 * @brief Output the credits message
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 * @param data with the pointer for the address were to store the data
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int generate_dataset_from_arguments(struct gengetopt_args_info args_info, int *data);

			/**
			 * @brief Generates a random data set of integers in the interval o MAX_INT
			 * @param data with the pointer for the address were to store the data
			 * @param n with the number of elements to generate
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int generate_dataset(int *data, int n);

			/**
			 * @brief Generates a random data set of integers in a specific interval
			 * @param data with the pointer for the address were to store the data
			 * @param n with the number of elements to generate
			 * @param minimum with lower limit
			 * @param maximum with upper limit
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int generate_dataset_in_interval(int *data, int n, int minimum, int maximum);

	#ifdef __cplusplus
		}
	#endif

#endif	/* MODULE_GENERATE_DATASET_H */

