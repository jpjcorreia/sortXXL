/**
 * @file aux.c
 * @brief source file for the auxiliary functions
 * @date 2012-10-22 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#include "../3rd/debug.h"
#include "constants.h"
#include "aux.h"

/**
 * @brief This function is based on the 3rd/debug.c debug function and operates using a very similar mode.
 * However, this function only outputs a debug message if the SHOW_DEBUG is enabled.
 * Unfortunately, by the information that we have found, it isn't possible to call another function with optional parameters without changing it.
 * (see http://c-faq.com/varargs/handoff.html for reference)
 *
 * @param file string with the filename that output the debug message
 * @param line integer with the line were this function was called
 * @param format string like the ones used with functions like "printf"
 * @param ... variable number of parameters
 * @see debug
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
void my_debug(const char* file, const int line, char* format, ...){
    // silence the warnings
    (void)file; (void)line; (void)format;

    // if we have the SHOW_DEBUG enabled, let's output the error
    #ifdef SHOW_DEBUG
        va_list argp;
        va_start(argp, format);
        fprintf(stderr, "[%s@%d] DEBUG - ", file, line);
        vfprintf(stderr, format, argp);
        va_end(argp);
        fprintf(stderr, "\n");
        fflush(stderr);
    #endif
}

/**
 * @brief Checks if a directory can be opened/exists
 * @param dirname to check
 * @return integer TRUE if the directory exists, FALSE otherwise
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
int directory_exists(char* dirname){
    DIR *dir = NULL;
    if((dir = opendir(dirname))!=NULL){
        closedir(dir);
        return TRUE;
    }
    return FALSE;
}

/**
 * @brief Checks if a file exists for the given mode
 * @param filename with the filename to check
 * @param mode with the mode to use
 * @return integer TRUE if the directory exists, FALSE otherwise
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
int file_exists(char* filename, char* mode){
    FILE *file = NULL;

    // Open the file
    if((file=fopen(filename,mode))!=NULL){
        fclose(file);
        return TRUE;
    }
    return FALSE;
}

/**
 * @brief Open a file descriptor for the file and redirects the stdout to this file
 * @param filename with the filename to insert the contents
 * @param attrib attributes to use in the fopen
 * @return FILE* descriptor or NULL on error
 * @see fopen
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
FILE* open_stdout_file(char* filename, char* attrib){
    FILE *file = NULL;
    // Create the file path and try to open the file
    if((file=fopen(filename, attrib))==NULL){
        ERROR(M_OPEN_STDOUT_FILE_FAILED,"\nUnable to open_stdout_file\n");
        return NULL;
    }else{
        // Close the stdout descriptor
        close(fileno(stdout));
        // Put the file descriptor to the position now free position left by closing the stdout
        dup(fileno(file));
    }
    return file;
}

/**
 * @brief Closes the file descriptor for the file and restore the stdout behavior
 * @param file FILE* with the file descriptor
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
void close_stdout_file(FILE* file){
    if(file!=NULL){
        fflush(stdout);
        // Restore the stdout behavior
        dup(fileno(stdout));
        // Close the file descriptor
        close(fileno(file));
        // Close the file descriptor
        fclose(file);
        clearerr(stdout);
        file=NULL;
    }
}

/**
 * @brief Create the parent directories (if they don't exist) for the dir or file path given
 *      (visit http://stackoverflow.com/a/9210960 for reference)
 * @param file_path char* with the file name
 * @param mode mode_t with the directory mode for the directory
 * @return M_OK if the path was created and/or exists, M_FAILED_MK_PATH on error
 *
 * @author Yaroslav Stavnichiy, Cláudio Esperança <cesperanc@gmail.com>
 */
int make_directory(char* file_path, mode_t mode) {
    #ifdef SHOW_DEBUG
        // Assert only on debug mode
        assert(file_path && *file_path);
    #endif

    char* p;
    
    for (p=strchr(file_path+1, '/'); p; p=strchr(p+1, '/')) {
        *p='\0';
        if (mkdir(file_path, mode)==-1) {
            if (errno!=EEXIST) { 
                *p='/';
                break;
            }
        }
        *p='/';
    }
    if(directory_exists(file_path)!=TRUE && mkdir(file_path, mode)==-1){
        ERROR(M_LOCALTIME_FAILED,"\nUnable to make_directory\n");
        return M_FAILED_MK_PATH; 
    }
    return M_OK;
}

/**
 * @brief Remove the directory and all of its contents
 *      (visit http://stackoverflow.com/a/2256974 for reference)
 * @param path char* with the directory to remove
 * @return int with the result
 *
 * @author asveikau
 */
int remove_directory(const char *path){
   DIR *d = opendir(path);
   size_t path_len = strlen(path);
   int r = -1;

   if (d){
      struct dirent *p;
      r = 0;
      while (!r && (p=readdir(d))){
          int r2 = -1;
          char *buf;
          size_t len;

          /* Skip the names "." and ".." as we don't want to recurse on them. */
          if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")){
             continue;
          }

          len = path_len + strlen(p->d_name) + 2; 
          buf = (char *)malloc(len);

          if (buf){
             struct stat statbuf;

             snprintf(buf, len, "%s/%s", path, p->d_name);
             if (!stat(buf, &statbuf)){
                if (S_ISDIR(statbuf.st_mode)){
                   r2 = remove_directory(buf);
                }else{
                   r2 = unlink(buf);
                }
             }
             free(buf);
          }
          r = r2;
      }
      closedir(d);
   }

   if (!r){
      r = rmdir(path);
   }
   return r;
}

/**
 * @brief Get the current time base on the given parameters
 * @param format to format the date
 * @param num_chars with the maximum number of characters of the string
 * @return string with the date in the given format
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
char* get_current_time(char* format, int num_chars){
    char* date = NULL;
    time_t t;
    struct tm *ltm;

    // Let's reserve one more char for the termination of the string
    num_chars++;

    t = time(NULL);
    ltm = localtime(&t);

    if((date = (char *)malloc((num_chars)*sizeof(char)))==NULL){
        ERROR(M_FAILED_MEMORY_ALLOCATION,"\nMemory allocation failed for date");
    }

    if (ltm == NULL) {
        ERROR(M_LOCALTIME_FAILED,"\nUnable to get the local time");
    }

    if (strftime(date, num_chars, format, ltm) == 0) {
        ERROR(M_FORMATTIME_FAILED,"\nUnable to format the date");
    }
    return date;
}

/**
 * @brief Concatenate the filename with a directory path
 * @param directory with the base path
 * @param filename with the filename string
 * @param separator with the char to use as separator of the directory and the filename
 * @return string with the full path for the file
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>
 */
char* concatenate_filename(const char* directory, const char* filename, char separator){
    char* result = NULL;
    size_t len;
    len = (strlen(directory)+strlen(filename)+2)*sizeof(char);
    if((result = (char *)malloc(len))==NULL){
        ERROR(M_FAILED_MEMORY_ALLOCATION,"\nMemory allocation failed for filename concatenation");
    }
    snprintf(result, len, "%s%c%s", directory, separator, filename);
    
    return result;
}

/**
 * @brief Concatenate the filename with a directory path
 *      (visit http://minkirri.apana.org.au/~abo/projects/librsync/small-1.0/replace/basename.c for original source code)
 * @param path with the full path
 * @param separator with the separator char
 * @return pointer of the basename (after the last separator char) or since the begining of the string
 *
 * @author Gary V. Vaughan
 */
char* base_name (char *path, const char separator){
    /* Search for the last directory separator in PATH.  */
    char *basename = strrchr (path, separator);

    /* If found, return the address of the following character, or the start of the parameter passed in.  */
    return basename ? ++basename : (char*)path;
}

/**
 * @brief Calculate the difference between two timeval in seconds
 * @param start with the start time
 * @param end with the end time
 * @return float with the difference
 *
 * @author Cláudio Esperança <cesperanc@gmail.com>, Diogo Serra <2081008@student.estg.ipleiria.pt>
 */
float time_diff(struct timeval start, struct timeval end){
	int seconds_divisor = 1000000;

	return ((float)(((end.tv_sec * seconds_divisor + end.tv_usec) - (start.tv_sec * seconds_divisor + start.tv_usec))))/seconds_divisor;
}

int get_next_power_of_two(int size){
	int n = 1;
	while (n < size) n *= 2;
	return n;
}

int get_previous_power_of_two(int size){
	int n = 1;
	while (n*2 < size) n *= 2;
	return n;
}

void pad_data_to_align_with(int *data, int current_size, int to_size, int with){
	int i;
	for(i=current_size; i<to_size; i++){
		data[i] = with;
	}
}

void pad_data_to_align(int *data, int current_size, int to_size){
	pad_data_to_align_with(data, current_size, to_size, INT_MAX);
}

int pad_data_to_align_with_next_power_of_two_with(int *data, int current_size, int with){
	int next_power_of_two = get_next_power_of_two(current_size);
	pad_data_to_align_with(data, current_size, next_power_of_two, with);
	return next_power_of_two;
}

int pad_data_to_align_with_next_power_of_two(int *data, int current_size){
	int next_power_of_two = get_next_power_of_two(current_size);
	pad_data_to_align(data, current_size, next_power_of_two);
	return next_power_of_two;
}
