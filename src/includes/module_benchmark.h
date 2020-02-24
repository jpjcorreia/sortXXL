/**
 * @file module_benchmark.h
 * @brief header file for the benchmark module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_BENCHMARK_H
	#define	MODULE_BENCHMARK_H
	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif

			#include "module_shared_stats.h"

			/** @brief statistical information parameters */
			typedef struct stats {
				float chronological_time; 		/**< @brief time interval in microseconds between the application start and results presentation */
				float gpu_time; 				/**< @brief GPU code execution time in microseconds */
				int total_sorted; 				/**< @brief number of elements sorted */
				int first_five_sorted[5]; 		/**< @brief first five sorted elements */
				int last_five_sorted[5]; 		/**< @brief last five sorted elements */
				int num_times; 					/**< @brief number of times to sort the data in benchmark mode */
				float min_time; 				/**< @brief minimum execution time for sorting the data in benchmark mode */
				float max_time; 				/**< @brief maximum execution time for sorting the data in benchmark mode */
				float avg_time; 				/**< @brief average execution time for sorting the data in benchmark mode */
				float std_time; 				/**< @brief standard deviation execution time for sorting the data */
			} STATS;

			/**
			 * @brief Flag the exit request
			 * @param signum with the signal request
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void termination_handler (int signum);

			/**
			 * @brief Generate the benchmark information
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 * @param stats with the reference to the statistical data
			 * @return exit code
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int benchmark(struct gengetopt_args_info args_info, SHARED_STATS_T* stats);

	#ifdef __cplusplus
		}
	#endif
#endif	/* MODULE_BENCHMARK_H */

