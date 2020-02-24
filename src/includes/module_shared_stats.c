/**
 * @file module_shared_stats.c
 * @brief source code file for the shared statistics module
 * @date 2012-12-30 File creation
 * @author Cláudio Esperança <2120917@my.ipleiria.pt>, Diogo Serra <2120915@my.ipleiria.pt>, João Correia <2111415@my.ipleiria.pt>
 */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>

#include "../3rd/debug.h"
#include "../3rd/semaforos.h"
#include "../3rd/sortXXL_cmd.h"

#include "constants.h"
#include "aux.h"
#include "module_shared_stats.h"

int shared_stats(SHARED_STATS_T** stats,  char* path_name, int number_of_benchmarks, int total_to_sort){
	int result = initialize_shared_stats(&(*stats), path_name, number_of_benchmarks, number_of_benchmarks);

	if(*stats!=NULL){
		// Test for errors
		switch(result){
			case M_SEMSET_FAILED_FOR_DATA:
				if((*stats)->elapsedTimes!=NULL)
					shmdt((*stats)->elapsedTimes);
					/* no break */
			case M_SHMAT_FAILED_FOR_DATA_STATS:
				if(((*stats)->shm_data_id)!=0)
					sem_delete((*stats)->shm_data_id);
				/* no break */
			case M_SEMCREATE_FAILED_FOR_DATA:
				if(((*stats)->shm_elapsed_times_data_id)!=0)
					shmctl((*stats)->shm_elapsed_times_data_id, IPC_RMID, 0);
				if(((*stats)->shm_data_id)!=0)
					shmctl((*stats)->shm_data_id, IPC_RMID, 0);
				/* no break */
			case M_SHMGET_FAILED_FOR_DATA:
			case M_FTOK_FAILED_FOR_DATA:
				if((*stats)->elapsedTimes!=NULL)
					shmdt((*stats)->elapsedTimes);
				if((*stats)!=NULL)
					shmdt((*stats));
				*stats=NULL;
				/* no break */
		}
	}
	return result;
}

int initialize_shared_stats(SHARED_STATS_T** stats,  char* path_name, int number_of_benchmarks, int total_to_sort){
	key_t data_key, elapsed_times_data_key;		/**< @brief store a System V IPC key for the data */
	int shm_data_id;
	SHARED_STATS_T* shared_stats;

	*stats=NULL;

	// Generate a shared memory key for the data key
	if((data_key = ftok(path_name, 'd'))==-1){
		MY_DEBUG("\nShared memory key generation for the data failed.\n");
		return M_FTOK_FAILED_FOR_DATA;
	}
	// Creates a shared memory zone for the data
	if((shm_data_id = shmget(data_key, sizeof(SHARED_STATS_T), 0600|IPC_CREAT|IPC_EXCL))==-1){
		MY_DEBUG("\nError while creating the shared memory for the data with the key [%x]. Is another %s running? If not, remove the shared memory segment(s) using 'ipcs' and 'ipcrm -M'\n", data_key, path_name);
		return M_SHMGET_FAILED_FOR_DATA;
	}
	// Attach shared memory for the data to this process
	if((shared_stats = (SHARED_STATS_T*) shmat(shm_data_id, NULL, 0))==(void *)-1){
		MY_DEBUG("\nError while attaching on the shared memory segment for the data\n");
		return M_SHMAT_FAILED_FOR_DATA_STATS;
	}
	// Creates the semaphore for the shared memory data access
	if((shared_stats->sem_data_id = sem_create(data_key, 2, 0600|IPC_CREAT|IPC_EXCL))==-1){
		MY_DEBUG("\nError while creating the semaphore for the access to the shared memory data with the key [%x]. Is another %s running? If not, remove the shared memory segment(s) using 'ipcs' and 'ipcrm -S'\n", data_key, path_name);
		return M_SEMCREATE_FAILED_FOR_DATA;
	}
	// Initializes the semaphore for the access to the data
	if(sem_setvalue(shared_stats->sem_data_id, MUTEX_DATA_ACCESS, 1)==-1){
		MY_DEBUG("\nError while setting the value for the semaphore\n");
		return M_SEMSET_FAILED_FOR_DATA;
	}
	// Initializes the semaphore for the data release
	if(sem_setvalue(shared_stats->sem_data_id, MUTEX_DATA_AVAILABLE, 0)==-1){
		MY_DEBUG("\nError while setting the value for the semaphore\n");
		return M_SEMSET_FAILED_FOR_DATA;
	}


	// Generate a shared memory key for the elapsed times data key
	if((elapsed_times_data_key = ftok(path_name, 'e'))==-1){
		MY_DEBUG("\nShared memory key generation for the data failed.\n");
		return M_FTOK_FAILED_FOR_DATA;
	}
	// Creates a shared memory zone for the elapsed times data
	if((shared_stats->shm_elapsed_times_data_id = shmget(elapsed_times_data_key, sizeof(float)*number_of_benchmarks, 0600|IPC_CREAT|IPC_EXCL))==-1){
		MY_DEBUG("\nError while creating the shared memory for the elapsed times data with the key [%x]. Is another %s running? If not, remove the shared memory segment(s)\n", elapsed_times_data_key, path_name);
		return M_SHMGET_FAILED_FOR_DATA;
	}
	// Attach shared memory for the elapsed times data to this process
	if((shared_stats->elapsedTimes = (float *) shmat(shared_stats->shm_elapsed_times_data_id, NULL, 0))==(void *)-1){
		MY_DEBUG("\nError while attaching on the shared memory segment for the elapsed times data\n");
		return M_SHMAT_FAILED_FOR_DATA_STATS;
	}

	shared_stats->locked = FALSE;
	shared_stats->current_time = 0;
	shared_stats->total_to_sort = total_to_sort;
	shared_stats->total_times = number_of_benchmarks;

	shared_stats->shm_data_id = shm_data_id;

	*stats = shared_stats;

	for(total_to_sort=0; total_to_sort<number_of_benchmarks; total_to_sort++){
		shared_stats->elapsedTimes[total_to_sort] = total_to_sort;
	}

	return EXIT_SUCCESS;
}

void lock_shared_stats_for_access(SHARED_STATS_T* stats){
	if(stats!=NULL){
		if(sem_down(stats->sem_data_id, MUTEX_DATA_ACCESS)==-1){
			ERROR(M_SEMDOWN_FAILED, "\nError while locking a semaphore\n");
		}
	}
}

void release_shared_stats_for_access(SHARED_STATS_T* stats){
	if(stats!=NULL){
		if(sem_up(stats->sem_data_id, MUTEX_DATA_ACCESS)==-1){
			ERROR(M_SEMUP_FAILED, "\nError while releasing a semaphore\n");
		}
	}
}

void release_new_shared_stat_data(SHARED_STATS_T* stats){
	int semvalue=0;
	if(stats!=NULL){
		if((semvalue=sem_getvalue(stats->sem_data_id, MUTEX_DATA_AVAILABLE))<0){
			ERROR(M_SEMGETVALUE_FAILED, "\nError while getting a semaphore value\n");
		}
		if(semvalue==0){
			if(sem_up(stats->sem_data_id, MUTEX_DATA_AVAILABLE)==-1){
				ERROR(M_SEMUP_FAILED, "\nError while releasing a semaphore\n");
			}
		}
	}
}

void wait_for_new_shared_stat_data(SHARED_STATS_T* stats){
	if(stats!=NULL){
		if(sem_down(stats->sem_data_id, MUTEX_DATA_AVAILABLE)==-1){
			ERROR(M_SEMDOWN_FAILED, "\nError while locking a semaphore");
		}
	}
}

int remove_shared_stats(SHARED_STATS_T* stats){
	int shm_data_id;

	if(stats!=NULL){
		shm_data_id = stats->shm_data_id;
		lock_shared_stats_for_access(stats);

		// Detach shared memory for the elapsed times
		if (shmdt(stats->elapsedTimes) == -1){
			ERROR(M_SHMDT_FAILED, "\nDetach shared memory for the elapsed times failed\n");
			return M_SHMDT_FAILED;
		}
		// Remove the shared memory segment for the elapsed times data
		if (shmctl(stats->shm_elapsed_times_data_id, IPC_RMID, 0) == -1){
			ERROR(M_SHMCTL_FAILED, "\nFailed to remove the shared memory segment for the elapsed times data\n");
			return M_SHMCTL_FAILED;
		}

		// Remove the semaphore for the data
		if (sem_delete(stats->sem_data_id) == -1){
			ERROR(M_SEMDELETE_FAILED, "\nFailed to remove the semaphore for the data\n");
			return M_SEMDELETE_FAILED;
		}

		// Detach shared memory for the data
		if (shmdt(stats) == -1){
			ERROR(M_SHMDT_FAILED, "\nFailed to detach shared memory for the data\n");
			return M_SHMDT_FAILED;
		}

		// Remove the shared memory segment for the data
		if (shmctl(shm_data_id, IPC_RMID, 0) == -1){
			ERROR(M_SHMCTL_FAILED, "\nFailed to remove the shared memory segment for the data\n");
			return M_SHMCTL_FAILED;
		}

		stats = NULL;

		MY_DEBUG("\nShared statistics removed\n");
	}

	return 0;
}

void append_stat(SHARED_STATS_T* stats, float elapsedTime, int currentTest){
	// Lock the data for access

	if(stats!=NULL){
		lock_shared_stats_for_access(stats);
		stats->current_time = currentTest;
		stats->elapsedTimes[currentTest] = elapsedTime;

		// If the data wasn't released yet, release it
		release_new_shared_stat_data(stats);

		// Release the data for access
		release_shared_stats_for_access(stats);
	}
}

int initialize_shared_stats_in_client(SHARED_STATS_T** stats, char* path_name){
        key_t data_key, elapsed_times_data_key; 		/**< @brief store a System V IPC key for the data */
        int shm_data_id;
        struct shmid_ds info;

        // Generate a shared memory key for the data key
		if((data_key = ftok(path_name, 'd'))==-1){
			MY_DEBUG("\nShared memory key generation for the data failed.\n");
			return M_FTOK_FAILED_FOR_DATA;
		}
		// Gets a shared memory zone for the data
		if((shm_data_id = shmget(data_key, 0, 0))==-1){
			MY_DEBUG("\nError while getting the shared memory for the data with the key [%x].\n", data_key);
			return M_SHMGET_FAILED_FOR_DATA;
		}

        info.shm_nattch = 0; // to ignore the uninitialized warning
        if (shmctl(shm_data_id, IPC_STAT, &info) == -1){
			MY_DEBUG("\nError while getting information about the memory segment.\n", data_key, path_name);
			return M_SHMCTL_FAILED;
        }else{
			switch(info.shm_nattch){
				case 0:
					printf("\nThere is no program attached to the shared memory (Is the %s running?)\n", path_name);
					return M_NO_PROGRAM_ATTACHED_TO_MEMORY;
				case 1:
					break;
				default:
					MY_DEBUG("\nAnother process (%d) is already listening\n", info.shm_lpid);
					return M_MAXIMUM_SHMAT_CLIENT_CONNECTED;
			}
        }
        // Attach shared memory for the control to this process
        if((*stats = (SHARED_STATS_T *) shmat(shm_data_id, NULL, 0))==(void *)-1){
			MY_DEBUG("\nError while attaching on the shared memory segment for the statistical data\n");
			return M_SHMAT_FAILED_FOR_DATA_STATS;
        }
        // Associate the semaphore for the shared memory zone with the controller
        if(((*stats)->sem_data_id = semget(data_key, 0, 0))==-1){
			MY_DEBUG("\nError while getting the semaphore for the access to the shared memory control data. Is the %s running?", path_name);
			return M_SEMGET_FAILED_FOR_DATA;
        }

        (*stats)->shm_data_id = shm_data_id;

        // Generate a shared memory key for the elapsed times data key
		if((elapsed_times_data_key = ftok(path_name, 'e'))==-1){
			MY_DEBUG("\nShared memory key generation for the data failed.\n");
			return M_FTOK_FAILED_FOR_DATA;
		}
		// Gets a shared memory zone for the data
		if(((*stats)->shm_elapsed_times_data_id = shmget(elapsed_times_data_key, 0, 0))==-1){
			MY_DEBUG("\nError while getting the shared memory for the data with the key [%x].\n", elapsed_times_data_key);
			return M_SHMGET_FAILED_FOR_DATA;
		}

        // Attach shared memory for the control to this process
		if(((*stats)->elapsedTimes = (float *) shmat((*stats)->shm_elapsed_times_data_id, NULL, 0))==(void *)-1){
			MY_DEBUG("\nError while attaching on the shared memory segment for the statistical data\n");
			return M_SHMAT_FAILED_FOR_DATA_STATS;
		}

        return EXIT_SUCCESS;
}

int remove_shared_stats_in_client(SHARED_STATS_T* stats){
	// Update the lock
	lock_shared_stats_for_access(stats);
	stats->locked=FALSE;
	release_shared_stats_for_access(stats);

	//controller_stat->control_data->stats = NULL;
	// Detach from the shared memory
	if (shmdt(stats->elapsedTimes) == -1){
		return M_SHMDT_FAILED;
	}

	// Detach from the shared memory
	if (shmdt(stats) == -1){
		return M_SHMDT_FAILED;
	}
	stats=NULL;

	return EXIT_SUCCESS;
}
