/**
 * @file aux.h
 * @brief header file for the auxiliary functions
 * @date 2012-10-22 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef AUX_H_
	#define AUX_H_

	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif
			// defines
			/**
			 * @brief Macro to print on the stderr useful depuration information.
			 * It accepts a variable number of parameters
			 *
			 * @see my_debug() for more information about this function
			 */
			#define MY_DEBUG(...) my_debug(__FILE__, __LINE__,__VA_ARGS__)

			// prototypes
			void my_debug(const char*, const int, char*, ...);

			int directory_exists(char*);
			int file_exists(char*, char*);

			FILE* open_stdout_file(char*, char*);
			void close_stdout_file(FILE*);
			int make_directory(char*, mode_t);
			int remove_directory(const char *);
			char* get_current_time(char*, int);
			char* concatenate_filename(const char*, const char*, const char);
			char* base_name (char *, const char);
			float time_diff(struct timeval start, struct timeval end);

			/**
			 * @brief Get the next power of two greater than size
			 * @param size with the current size
			 * @return integer with the next value
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int get_next_power_of_two(int size);

			/**
			 * @brief Get the previous power of two of the provided number
			 * @param size with the current size
			 * @return integer with the immediate previous value
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int get_previous_power_of_two(int size);

			/**
			 * @brief Align the data with a specific size using a padding
			 * @param data with the array to store the data
			 * @param current_size with the current size of the array without padding
			 * @param to_size with final size of the array
			 * @param with with the value to use as a padding
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void pad_data_to_align_with(int *data, int current_size, int to_size, int with);

			/**
			 * @brief Align the data with a specific size using 0 as padding
			 * @param data with the array to store the data
			 * @param current_size with the current size of the array without padding
			 * @param to_size with final size of the array
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void pad_data_to_align(int *data, int current_size, int to_size);

			/**
			 * @brief Align the data with a power of two size using a padding
			 * @param data with the array to store the data
			 * @param current_size with the current size of the array without padding
			 * @param with with the value to use as a padding
			 * @return integer with the final array size
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size, int with);

			/**
			 * @brief Align the data with a power of two size using 0 as padding
			 * @param data with the array to store the data
			 * @param current_size with the current size of the array without padding
			 * @return integer with the final array size
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int pad_data_to_align_with_next_power_of_two(int *data, int current_size);

	#ifdef __cplusplus
		}
	#endif
#endif /* AUX_H_ */
