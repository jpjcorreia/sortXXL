/**
 * @file module_file_dataset.c
 * @brief source code file for generate a data set from a file
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <string.h>

#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "module_file_dataset.h"

int file_dataset(struct gengetopt_args_info args_info, int **data, int *n){
	int minimum = 0, maximum = INT_MAX, ps, *work_data;
	size_t size=0;

	char buff[BUFSIZ], *pos, *buffCopy;

	FILE *read;

	if (args_info.min_given == 1){
		minimum = args_info.min_arg;
	}

	if (args_info.max_given == 1){
		maximum = args_info.max_arg;
	}

	if(args_info.input_given == 1){
		// Open the file
		if((read=fopen(args_info.input_arg,"r"))==NULL) {
			printf("Can't open the file %s\n", args_info.input_arg);
			return EXIT_FAILURE;
		}

		// Read line by line
		while(fgets(buff, sizeof(buff), read)!=NULL){

			// If the first character is #, ignored it
			if (buff[0] == '#')
				continue;

			// Get the first group of characters until a space is found
			buffCopy = strdup(buff);
			pos = strtok(buffCopy, " ");

			do{
				// If a number is found
				if (isdigit(*pos)){
					// Convert and store it (if between allowed interval)
					ps=atoi(pos);

					if(ps>=minimum && ps<=maximum){
						// Increase data store capacity by 1000 at a time
						if(*n*sizeof(int)<=size){
							size = (*n+1000)*sizeof(int);
							work_data = (int*) realloc(*data, size);
							if (work_data!=NULL) {
								*data = work_data;
							}else{
								printf("Problem while allocating the necessary memory to load the data set (memory exhausted?)\n");
								return EXIT_FAILURE;
							}
						}
						work_data[(*n)++]=ps;
					}
				}
				// Get to next chunk
				pos=strtok(NULL, " ");
			}while(pos && *pos!='\n');

			free(buffCopy);
		}

		// Remove the last unused slot
		(*n)--;

		// Close the file
		fclose(read);

		// Check if any data was loaded
		if(*n<=0){
			printf("No valid data found in the file. Are you sure that numbers exist on that file? \n");
			return EXIT_FAILURE;
		}

		// Shrink the data store to the necessary size
		if(*n*sizeof(int)>size){
			size = *n*sizeof(int);
			work_data = (int*) realloc(*data, size);
			if (work_data!=NULL) {
				*data = work_data;
			}else{
				printf("Problem while freeing the unused memory\n");
				return EXIT_FAILURE;
			}
		}

		return EXIT_SUCCESS;
	}
	return EXIT_FAILURE;
}
