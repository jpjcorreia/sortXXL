/**
 * @file constants.h
 * @brief header file for constants
 * @date 2012-10-22 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef SORTXXL_CONSTANTS_H_
	#define SORTXXL_CONSTANTS_H_

	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif

		/**
		 * A default type for a boolean false value
		 */
		#define FALSE 0

		/**
		 * A default type for a boolean true value
		 */
		#define TRUE 1

		// mutexes
		/**
		 * Mutex constant for the statistical control semaphore
		 */
		#define MUTEX_CONTROL_STATS 0

		/**
		 * Mutex constant for the statistical control semaphore
		 */
		#define MUTEX_EXIT 1

		/**
		 * Mutex constant for the data available semaphore
		 */
		#define MUTEX_DATA_AVAILABLE 0

		/**
		 * Mutex constant for the data ACCESS semaphore
		 */
		#define MUTEX_DATA_ACCESS 1

		// exit constants
		/**
		 * Define the default value for the OK message
		 */
		#define M_OK 0
		/**
		 * Define the exit value for the invalid parameters message
		 */
		#define M_INVALID_PARAMETERS 1

		/**
		 * Define the exit value for the make path fail message
		 */
		#define M_FAILED_MK_PATH 2

		/**
		 * Define the exit value for the remove directory fail message
		 */
		#define M_FAILED_REMOVE_DIRECTORY 3

		/**
		 * Define the exit value for the memory allocation fail message
		 */
		#define M_FAILED_MEMORY_ALLOCATION 4

		/**
		 * Define the exit value for the localtime fail message
		 */
		#define M_LOCALTIME_FAILED 5

		/**
		 * Define the exit value for the format time fail message
		 */
		#define M_FORMATTIME_FAILED 6

		/**
		 * Define the exit value for the open_stdout_file fail message
		 */
		#define M_OPEN_STDOUT_FILE_FAILED 7

		/**
		 * Define the exit value for the output_directory fail message
		 */
		#define M_OUTPUT_DIRECTORY_FAILED 8

		/**
		 * Define the exit value for the strcpy fail message
		 */
		#define M_FAILED_STRCPY 9

		/**
		 * Define the exit value for the fork failed message
		 */
		#define M_FORK_FAILED 10

		/**
		 * Define the exit value for the sigaction sigint error
		 */
		#define M_SIGACTION_SIGINT_FAILED 11

		/**
		 * Define the exit value for the ftok error for the control key
		 */
		#define M_FTOK_FAILED_FOR_CONTROL_KEY 12

		/**
		 * Define the exit value for the shared memory allocation error for the controller
		 */
		#define M_SHMGET_FAILED_FOR_CONTROL_ID 13

		/**
		 * Define the exit value for the semaphore creation error for the controller
		 */
		#define M_SEMCREATE_FAILED_FOR_CONTROL 14

		/**
		 * Define the exit value on error while setting the value of a control semaphore
		 */
		#define M_SEMSET_FAILED_FOR_CONTROL 15

		/**
		 * Define the exit value for the shared memory attach error for the control
		 */
		#define M_SHMAT_FAILED_FOR_CONTROL 16

		/**
		 * Define the exit value on error while getting a semaphore for the control
		 */
		#define M_SEMGET_FAILED_FOR_CONTROL 17

		/**
		 * Define the exit value for the ftok error
		 */
		#define M_FTOK_FAILED_FOR_DATA 18

		/**
		 * Define the exit value for the shared memory allocation error
		 */
		#define M_SHMGET_FAILED_FOR_DATA 19

		/**
		 * Define the exit value for the semaphore creation error
		 */
		#define M_SEMCREATE_FAILED_FOR_DATA 20

		/**
		 * Define the exit value for the shared memory attach error
		 */
		#define M_SHMAT_FAILED_FOR_DATA_STATS 21

		/**
		 * Define the exit value on error while setting the value of a semaphore
		 */
		#define M_SEMSET_FAILED_FOR_DATA 22

		/**
		 * Define the exit value on error while locking a semaphore
		 */
		#define M_SEMDOWN_FAILED 23

		/**
		 * Define the exit value on error while releasing a semaphore
		 */
		#define M_SEMUP_FAILED 24

		/**
		 * Define the exit value for the shared memory detach error
		 */
		#define M_SHMDT_FAILED 25

		/**
		 * Define the exit value for the shared memory control error
		 */
		#define M_SHMCTL_FAILED 26

		/**
		 * Define the exit value for the semaphore deletion error
		 */
		#define M_SEMDELETE_FAILED 27

		/**
		 * Define the exit value on error while getting a semaphore
		 */
		#define M_SEMGET_FAILED_FOR_DATA 28

		/**
		 * Define the exit value on error while getting the value of a semaphore
		 */
		#define M_SEMGETVALUE_FAILED 29

		/**
		 * Define the exit value when there is no program attached to the shared memory segment
		 */
		#define M_NO_PROGRAM_ATTACHED_TO_MEMORY 30

		/**
		 * Define the exit value when there the maximum number of shared memory client are connnected
		 */
		#define M_MAXIMUM_SHMAT_CLIENT_CONNECTED 31

		/**
		 * Define the exit value when there the GPU doesn't have the necessary compute capabilities
		 */
		#define UNSUFFICENT_GPU_CAMPABILITIES 32


	#ifdef __cplusplus
		}
	#endif
#endif /* SORTXXL_CONSTANTS_H_ */
