/**
 * @file module_about.h
 * @brief header file for the credits module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_ABOUT_H
	#define	MODULE_ABOUT_H

	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif

			/**
			 * @brief Output the credits message
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 *
			 * @author Joao Correia <joao.pedro.j.correia@gmail.com>
			 */
			void about_sort_XXL(struct gengetopt_args_info args_info);

	#ifdef __cplusplus
		}
	#endif
#endif	/* MODULE_ABOUT_H */

