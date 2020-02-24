#ifdef __cplusplus
	extern "C" {
		#include <cstddef>
#endif
		int lws_b64_encode_string(const char *in, int in_len, char *out, int out_size);
		/*
		 * returns length of decoded string in out, or -1 if out was too small
		 * according to out_size
		 */

		int
		lws_b64_decode_string(const char *in, char *out, int out_size);
		int
		lws_b64_selftest(void);
#ifdef __cplusplus
	}
#endif
