/**
 * @file module_system_info.h
 * @brief header file for the system information module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_SYSTEM_INFO_H
	#define	MODULE_SYSTEM_INFO_H

	#ifdef __cplusplus
		extern "C" {
	#include <cstddef>
	#endif

			/**
			 * @brief Output the CUDA system information
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 * @author Diogo Serra <2120915@my.ipleiria.pt>
			 */
			void system_info(struct gengetopt_args_info args_info);

			/**
			 * @brief Compute the number of cores string
			 * @author Diogo Serra <2120915@my.ipleiria.pt>
			 */
			char* number_of_cores(double, double);

	#ifdef __cplusplus
		}
	#endif

#endif	/* MODULE_SYSTEM_INFO_H */

