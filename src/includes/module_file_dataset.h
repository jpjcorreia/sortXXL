/**
 * @file module_file_dataset.h
 * @brief header file for generate a data set from a file
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_FILE_DATASET_H
	#define	MODULE_FILE_DATASET_H

	#ifdef __cplusplus
		extern "C" {
	#include <cstddef>
	#endif

			/**
			 * @brief Load the data set from a file
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 * @param data with the pointer for the address were to store the data
			 * @param n with the number of elements loaded
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int file_dataset(struct gengetopt_args_info args_info, int **data, int *n);

	#ifdef __cplusplus
		}
	#endif
#endif	/* MODULE_FILE_DATASET_H */

