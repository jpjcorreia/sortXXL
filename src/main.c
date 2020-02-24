/**
* \mainpage
* The sortXXL is an application to sort a big amount of numbers
* 
* 
* @file main.c
* @brief Main source file for the sortXLL program
* @date 2012-12-03 File creation
* @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "3rd/debug.h"
#include "3rd/sortXXL_cmd.h"

#include "includes/constants.h"
#include "includes/aux.h"

#include "includes/module_about.h"
#include "includes/module_system_info.h"
#include "includes/module_benchmark.h"
#include "includes/module_demo.h"
#include "includes/module_shared_stats.h"

#include "main.h"

int main(int argc, char *argv[]){

    /* Variable declarations */
    struct gengetopt_args_info args_info;// structure for the command line parameters processing
    SHARED_STATS_T* stats=NULL;
    int result = EXIT_SUCCESS;
    
    // Initializes the command line parser and check for the application parameters
    if (cmdline_parser(argc,argv,&args_info) != 0){
        DEBUG("\nInvalid parameters");
        result = M_INVALID_PARAMETERS;
    }

    if(result==EXIT_SUCCESS){

    	// Initialize the data structures for statistics sharing
    	shared_stats(&stats, argv[0], args_info.benchmark_given == 1?args_info.benchmark_arg:1, 0);

		// Run the demo
		demo(args_info, stats);

		// Sort the data
		benchmark(args_info, stats);

		// Output the system information
		system_info(args_info);

		// Output the credits
		about_sort_XXL(args_info);

		// Wait for the child processes to exit
		wait(&result);

		// Remove shared stats
		remove_shared_stats(stats);
    }
    // Free the command line parser memory
    cmdline_parser_free(&args_info);

    return result;
}
