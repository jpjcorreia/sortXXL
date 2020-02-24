/**
 * @file module_shared_stats.h
 * @brief header file for the shared statistics module
 * @date 2012-12-30 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_SHARED_STATS_H
	#define	MODULE_SHARED_STATS_H

	#ifdef __cplusplus
		extern "C" {
	#include <cstddef>
	#endif
			/** @brief statistical information parameters */
			typedef struct shared_stats {
				float *elapsedTimes; 			/**< @brief time interval in microseconds between the application start and results presentation */
				int total_to_sort; 				/**< @brief number of elements to sort */
				int total_times;				/**< @brief number of times to sort the data in benchmark mode */
				int current_time;				/**< @brief current test index */

				int locked;                     /**< @brief control if something is listening is listening */
				int shm_elapsed_times_data_id;  /**< @brief reference to the shared memory for the control data */
				int shm_data_id;                /**< @brief reference to the shared memory for the data */
				int sem_data_id;                /**< @brief reference to the semaphore for the access to shared memory data */
			} SHARED_STATS_T;

			/**
			 * @brief Implement the mechanism to share statistics between processes
			 * @param stats with the statistical data structure to be initialized and shared
			 * @param path_name with the executable name to generate a unique key for IPC resource sharing
			 * @param number_of_benchmarks with the number of tests to be executed and reserve the necessary memory resources
			 * @param total_to_sort with the number of items to be sorted
			 * @return exit code
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int shared_stats(SHARED_STATS_T** stats,  char* path_name, int number_of_benchmarks, int total_to_sort);

			/**
			 * @brief Initializes the statistical data structure, creates the shared memory segments and respective mutexes
			 * @param stats with the statistical data structure to be initialized and shared
			 * @param path_name with the executable name to generate a unique key for IPC resource sharing
			 * @param number_of_benchmarks with the number of tests to be executed and reserve the necessary memory resources
			 * @param total_to_sort with the number of items to be sorted
			 * @return exit code 0 if the data was successfully initialized, the resulting error code otherwise
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int initialize_shared_stats(SHARED_STATS_T** stats,  char* path_name, int number_of_benchmarks, int total_to_sort);

			/**
			 * @brief Lock the access to the data
			 * @param stats with the reference to the statistical data to lock access to
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void lock_shared_stats_for_access(SHARED_STATS_T* stats);

			/**
			 * @brief Release the access for the data
			 * @param stats with the reference to the statistical data to allow access to
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void release_shared_stats_for_access(SHARED_STATS_T* stats);

			/**
			 * @brief Release the access for the newly appended data
			 * @param stats with the reference to the statistical data to allow access to
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void release_new_shared_stat_data(SHARED_STATS_T* stats);

			/**
			 * @brief Lock the access to the data until new data is received
			 * @param stats with the reference to lock while waiting for new data
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void wait_for_new_shared_stat_data(SHARED_STATS_T* stats);

			/**
			 * @brief Detach the statistical data structures from the shared memory and try to release that segment
			 * @param stats with the reference to the statistical data
			 * @return exit code 0 if the data was successfully detached, the resulting error code otherwise
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int remove_shared_stats(SHARED_STATS_T* stats);

			/**
			 * @brief Add new statistic data to the shared memory segment
			 * @param stats with the reference to the statistical data
			 * @param elapsedTime with the elapsed time to register
			 * @param currentTest with the current test number
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void append_stat(SHARED_STATS_T* stats, float elapsedTime, int currentTest);

			/**
			 * @brief Initializes the statistical data structure for access by other client
			 * @param stats with the statistical data structure to be initialized and shared
			 * @param path_name with the executable name to generate a unique key for IPC resource sharing
			 * @return exit code 0 if the data was successfully initialized, the resulting error code otherwise
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int initialize_shared_stats_in_client(SHARED_STATS_T** stats, char* path_name);

			/**
			 * @brief Detach the client statistical data structures from the shared memory
			 * @param stats with the reference to the statistical data
			 * @return exit code 0 if the data was successfully detached, the resulting error code otherwise
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int remove_shared_stats_in_client(SHARED_STATS_T* stats);

	#ifdef __cplusplus
		}
	#endif
#endif	/* MODULE_SHARED_STATS_H */

