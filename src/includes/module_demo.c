/**
 * @file module_demo.c
 * @brief source code file for demo functionality
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include "../3rd/sortXXL_cmd.h"
#include "../3rd/debug.h"
#include "../3rd/libwebsockets/libwebsockets.h"

#include "constants.h"
#include "aux.h"
#include "module_shared_stats.h"
#include "module_demo.h"

/**
 * @brief Define the base directory to look for files to serve using HTTP
 */
#define LOCAL_RESOURCE_PATH "./demoFiles"


/**
 * @brief list of supported protocols and callbacks
 */
struct libwebsocket_protocols prots[] = {
	/* first protocol must always be HTTP handler */

	{
		"http-only",		/* name */
		(callback_function *) callback_http,		/* callback */
		0			/* per_session_data_size */
	},
	{
		"sort-xxl-stats",
		(callback_function *) callback_sorter_xxl_stats,
		sizeof(struct per_session_data_sort_xxl_update),
	},
	{
		NULL, NULL, 0		/* End of list */
	}
};

/**
 * @brief store the socket context globally for correct resource deallocation
 */
struct libwebsocket_context *context=NULL;

/**
 * @brief store the statistical data structure globally for global use
 */
SHARED_STATS_T* st;

int demo(struct gengetopt_args_info args_info, SHARED_STATS_T* stats){
    int n = 0;
    const char *cert_path = NULL;
    const char *key_path = NULL;
    char buf[LWS_SEND_BUFFER_PRE_PADDING + 1024 +
					  LWS_SEND_BUFFER_POST_PADDING];
    int port = args_info.demo_arg;
    //int use_ssl = 0;
    int opts = 0;
    //char interface_name[128] = "";
    const char *interface = NULL;
    #ifdef LWS_NO_FORK
        unsigned int oldus = 0;
    #endif

	struct sigaction action; // structure for the signal handling

	if(args_info.demo_given==1){

		// Let's fork the main program
		switch(fork()){
			case -1:
				ERROR(M_FORK_FAILED, "Fork for the demo mode failed!\n");
				return EXIT_FAILURE;

			case 0: //Child proceeds

				break;

			default: // Return to the parent process
				// We want to ignore the exit request to allow the child processes to exit normally and the parent free all the used resources afterwards
				action.sa_handler = SIG_IGN;
				// Mask without signals - won't block the signals
				sigemptyset(&action.sa_mask);
				action.sa_flags = 0;
				// Recover the blocked calls
				action.sa_flags |= SA_RESTART;
				// Specify the signals that will be handled
				if(sigaction(SIGINT, &action, NULL) < 0){
					ERROR(M_SIGACTION_SIGINT_FAILED, "Action for signal SIGINT failed!\n");
				}
				return EXIT_SUCCESS;
		}
		st = stats;

        if(stats==NULL){
        	MY_DEBUG("Unable to access statistic information\n");
			return EXIT_FAILURE;
        }

		register_signal_handlers();

        /*//Moved to the about module
        fprintf(stderr, "libwebsockets test server\n"
                "(C) Copyright 2010-2011 Andy Green <andy@warmcat.com> "
                "licensed under LGPL2.1\n");
        */
        //if (!use_ssl)
        //    cert_path = key_path = NULL;
        
        context = libwebsocket_create_context(port, interface, prots,
                                              libwebsocket_internal_extensions,
                                              cert_path, key_path, -1, -1, opts, NULL);
        if (context == NULL) {
        	MY_DEBUG("libwebsocket init failed\n");
            return EXIT_FAILURE;
        }

        /*
         * This example shows how to work with the forked websocket service loop
         */

        MY_DEBUG(" Using forked service loop\n");

        /*
         * This forks the websocket service action into a subprocess so we
         * don't have to take care about it.
         */

        n = libwebsockets_fork_service_loop(context);
        if (n < 0) {
            MY_DEBUG("Unable to fork service loop %d\n", n);
            return EXIT_FAILURE;
        }
        buf[LWS_SEND_BUFFER_PRE_PADDING] = 'x';

        // Wait for data
        while (context!=NULL) {
			wait_for_new_shared_stat_data(stats);
			libwebsockets_broadcast(&prots[PROTOCOL_SORT_XXL], &buf[LWS_SEND_BUFFER_PRE_PADDING], 1);
		}
        
        libwebsocket_context_destroy(context);
        
        //return 0;
        
	}
	return EXIT_SUCCESS;
}

void handle_signal(int signal) {
	int aux;

	/* Copy the value of the global variable errno */
	aux = errno;
	if(signal==SIGINT && context!=NULL){
		libwebsocket_context_destroy(context);
		MY_DEBUG("\nTerminating demo process\n");
		context=NULL;
	}
	/* Restore the value for the global variable errno */
	errno = aux;

	exit(EXIT_SUCCESS);
}

void register_signal_handlers(void){
	struct sigaction action; // structure for the signal handling

	// Signal handling function
	action.sa_handler = handle_signal;
	// Mask without signals - won't block the signals
	sigemptyset(&action.sa_mask);
	action.sa_flags = 0;
	// Recover the blocked calls
	action.sa_flags |= SA_RESTART;
	// Specify the signals that will be handled
	if(sigaction(SIGINT, &action, NULL) < 0){
		ERROR(M_SIGACTION_SIGINT_FAILED, "Action for signal SIGINT failed!\n");
	}
}




/* this protocol server (always the first one) just knows how to do HTTP */
int callback_http(struct libwebsocket_context *context,
                  struct libwebsocket *wsi,
                  enum libwebsocket_callback_reasons reason, void *user,
                  char *in,
                  //void *in,
                  size_t len)
{
	char client_name[128];
	char client_ip[128];
	switch (reason) {
        case LWS_CALLBACK_HTTP:
        	MY_DEBUG("serving HTTP URI %s\n", (char *)in);
            if (in && strcmp(in, "/favicon.ico") == 0) {
                if (libwebsockets_serve_http_file(wsi, LOCAL_RESOURCE_PATH"/favicon.ico", "image/x-icon"))
                	MY_DEBUG("Failed to send favicon\n");
                break;
            }
            
            if (in && strcmp(in, "/jquery-1.8.3.min.js") == 0) {
				if (libwebsockets_serve_http_file(wsi, LOCAL_RESOURCE_PATH"/jquery-1.8.3.min.js", "text/javascript"))
					MY_DEBUG("Failed to send jquery-1.8.3.min.js\n");
				break;
			}

            /* send the script... when it runs it'll start websockets */
            
            if (libwebsockets_serve_http_file(wsi, LOCAL_RESOURCE_PATH"/index.html", "text/html"))
            	MY_DEBUG("Failed to send HTTP file\n");
            break;
            
            /*
             * callback for confirming to continue with client IP appear in
             * protocol 0 callback since no websocket protocol has been agreed
             * yet.  You can just ignore this if you won't filter on client IP
             * since the default uhandled callback return is 0 meaning let the
             * connection continue.
             */
            
        case LWS_CALLBACK_FILTER_NETWORK_CONNECTION:
            
            libwebsockets_get_peer_addresses((int)(long)user, client_name,
                                             sizeof(client_name), client_ip, sizeof(client_ip));
            
            MY_DEBUG("Received network connect from %s (%s)\n", client_name, client_ip);
            
            /* if we returned non-zero from here, we kill the connection */
            break;
            
        default:
            break;
	}
    
	return 0;
}

int callback_sorter_xxl_stats(struct libwebsocket_context *context, struct libwebsocket *wsi, enum libwebsocket_callback_reasons reason, struct per_session_data_sort_xxl_update *data, char *in, size_t len){
	int n, i;
	char buf[LWS_SEND_BUFFER_PRE_PADDING + 512 + LWS_SEND_BUFFER_POST_PADDING];
	char *p = &buf[LWS_SEND_BUFFER_PRE_PADDING];

	switch (reason) {
		case LWS_CALLBACK_ESTABLISHED:
			MY_DEBUG("callback_sorter_xxl_stats: LWS_CALLBACK_ESTABLISHED\n");

			if(st!=NULL){
				lock_shared_stats_for_access(st);
				for(i=0;i<=st->current_time;i++){
					n = sprintf((char *)p, "{\"elapsedTime\": %f, \"currentTest\": %d, \"numberOfTests\": %d, \"numberOfElementsToSort\": %d}", st->elapsedTimes[i], i, st->total_times, st->total_to_sort);
					n = libwebsocket_write(wsi, p, n, LWS_WRITE_TEXT);
					if (n < 0) {
						MY_DEBUG("ERROR writing to socket");
						return EXIT_FAILURE;
					}
				}

				release_shared_stats_for_access(st);
			}
		break;

		 /*
		  * in this protocol, we just use the broadcast action as the chance to
		  * send our own connection-specific data and ignore the broadcast info
		  * that is available in the 'in' parameter
		  */
		case LWS_CALLBACK_BROADCAST:
			if(st!=NULL){
				lock_shared_stats_for_access(st);
				n = sprintf((char *)p, "{\"elapsedTime\": %f, \"currentTest\": %d, \"numberOfTests\": %d, \"numberOfElementsToSort\": %d}", st->elapsedTimes[st->current_time], st->current_time, st->total_times, st->total_to_sort);
				n = libwebsocket_write(wsi, p, n, LWS_WRITE_TEXT);
				if (n < 0) {
					MY_DEBUG("ERROR writing to socket");
					return EXIT_FAILURE;
				}

	    		release_shared_stats_for_access(st);
			}
			break;

		case LWS_CALLBACK_RECEIVE:
			MY_DEBUG("received %d\n", (int)len);
			if (len < 6)
				break;
			if (strcmp(in, "reset\n") == 0)
				data->time = 0;
			break;

			default:
				break;
	}

	return 0;
}
