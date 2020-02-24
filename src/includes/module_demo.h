/**
 * @file module_demo.h
 * @brief header file for the demo module
 * @date 2012-12-25 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */

#ifndef MODULE_DEMO_H
	#define	MODULE_DEMO_H

	#ifdef __cplusplus
		extern "C" {
			#include <cstddef>
	#endif
			#include "../3rd/libwebsockets/libwebsockets.h"
			#include "module_shared_stats.h"

			static int close_testing;

			/**
			 * @brief enum of supported protocols
			 */
			enum demo_protocols {
				/* always first */
				PROTOCOL_HTTP = 0,

				PROTOCOL_SORT_XXL,

				/* always last */
				DEMO_PROTOCOL_COUNT
			};

			/**
			 * @brief One of these is auto-created for each connection and a pointer to the
			 * appropriate instance is passed to the callback in the user parameter
			 *
			 * for this example protocol we use it to individualize the count for each
			 * connection.
			 */
			struct per_session_data_sort_xxl_update {
				float time; /**< @brief time interval in microseconds */
			};

			/**
			 * @brief Initializes the demo functionality
			 * @param args_info struct gengetopt_args_info with the parameters given to the application
			 * @param stats with the reference to the statistical data
			 * @return exit code
			 *
			 * @author Diogo Serra <2120915@my.ipleiria.pt>
			 */
			int demo(struct gengetopt_args_info args_info, SHARED_STATS_T* stats);

			/**
			 * @brief Handle the signals sent to the application
			 * @param signal integer with the signal to be handled
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void handle_signal(int signal);

			/**
			 * @brief Register the signal handler function the terminate the subprocess
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			void register_signal_handlers(void);

			/**
			 * @brief Handle web socket connections to the shared statistical data
			 * @param context with the libwebsocket_context
			 * @param wsi with the libwebsocket
			 * @param reason with the libwebsocket_callback_reasons
			 * @param data with the per_session_data_sort_xxl_update to send to the socket
			 * @param in with the data received
			 * @param len with the size
			 * @return exit code
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>
			 */
			int callback_sorter_xxl_stats(struct libwebsocket_context *context, struct libwebsocket *wsi, enum libwebsocket_callback_reasons reason, struct per_session_data_sort_xxl_update *data, char *in, size_t len);

			/**
			 * @brief Handle web socket connections to the shared statistical data
			 * @param context with the libwebsocket_context
			 * @param wsi with the libwebsocket
			 * @param reason with the libwebsocket_callback_reasons
			 * @param user with the data to send to the socket
			 * @param in with the data received
			 * @param len with the size
			 * @return exit code
			 *
			 * @author Cláudio Esperança <cesperanc@gmail.com>, Diogo Serra <2120915@my.ipleiria.pt>
			 */
			int callback_http(struct libwebsocket_context *context,
							  struct libwebsocket *wsi,
							  enum libwebsocket_callback_reasons reason, void *user,
							  char *in,
							  size_t len);

	#ifdef __cplusplus
		}
	#endif
#endif	/* MODULE_DEMO_H */

