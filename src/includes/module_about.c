/**
 * @file module_about.c
 * @brief source code file for the credits function
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>

#include "../3rd/sortXXL_cmd.h"

#include "module_about.h"

void about_sort_XXL(struct gengetopt_args_info args_info){
	if (args_info.about_given == 1){
    	printf("\n                           _    _  _    _ _\n                      _   \\ \\  / /\\ \\  / / |\n      ___  ___   ____| |_  \\ \\/ /  \\ \\/ /| |\n     /___)/ _ \\ / ___)  _)  )  (    )  ( | |\n    |___ | |_| | |   | |__ / /\\ \\  / /\\ \\| |_____\n    (___/ \\___/|_|    \\___)_/  \\_\\/_/  \\_\\_______)\n\n");

    	printf ("\n\t\t 2º projeto de CAD-MEICM ESTG/IPLeiria 2012-13\n");
    	printf ("\nAuthors:");
    	printf ("\nCláudio Esperança\t <2120917@my.ipleiria.pt>\nDiogo Serra\t\t <2120915@my.ipleiria.pt>\nJoão Correia\t\t <2111415@my.ipleiria.pt> \n\n");

    	printf ("\nCredits:");
		printf ("\n\tBase CUDA bitonic algorithm implementation: \n\t\tAyushi Sinha, Providence College. \n\t\tSorting on CUDA - Annual Celebration of Student Scholarship and Creativity, \n\t\thttp://digitalcommons.providence.edu/student_scholarship/7/, \n\t\tSpring 2011 \n\t\tLicense: unspecified\n");
		printf ("\n\tLibwebsockets: \n\t\t(C) Copyright 2010-2011 Andy Green <andy@warmcat.com> \n\t\thttp://git.warmcat.com/cgi-bin/cgit/libwebsockets/\n\t\tLicense: licensed under LGPL2.1\n\n");

    }
}
