#ifndef _APPCTX_HPP_
#define _APPCTX_HPP_

#include <petsctao.h>
#include <unistd.h>
#include "timer.hpp"

typedef enum {LOSS_IID, LOSS_CRF} LossName;
extern const char *LossNames[];

typedef struct{
	PetscInt wBegin, wEnd;	// The range of words stored in my process (global index, from 0)
													//  A loop should enumerate from wBegin up to (wEnd - 1)
													// For training data (3438 words) on two processes, 
													//     the first process has wBegin = 0 and wEnd = 1719
													// and the second process has wBegin = 1719 and wEnd = 3438.
													// For test data (3439 words) on two processes
													//     the first process has wBegin = 0 and wEnd = 1719
													// and the second process has wBegin = 1719 and wEnd = 3439.
	PetscInt wGlobalCount;	// Total number of words of the entire training or test set
	PetscInt lBegin, lEnd;	// range of letters: from the first letter of the first local word 
													//	to one plus the last letter of the last local word
	PetscInt lLocalCount;		// Total number of local letters 
	PetscInt maxWordLength; // max #letters among all words
	PetscInt *wLen;					// wlen[i] = #letters in the i-th word (i is indexed globally).  
													// This array has wGlobalCount elements, i.e.
													//	it stores the length of all words in the entire training/test set
													//  NOT just the length of local set of words.
													// For the training set, all processes have this same array (3438 elements).
													// For the test set, all processes also has this array (3439 elements).
} SeqCtx;


// User-defined application context - contains data needed by the
// application-provided call-back routines that evaluate the function,
// and gradient.
typedef struct {

  PetscMPIInt rank;					// the rank (id) of my process
	PetscMPIInt size;					// the total number of processes (number following -n in mpirun)
  
  PetscInt nclasses;				// Number of letter classes.  = 26 in our project

  /// training dataset and its statistics
	char data_path[PETSC_MAX_PATH_LEN];  // filename of the training data
	Vec labels;								// y_train, the label for training
	Mat data;									// X_train, the data matrix for training
  PetscInt dim;							// Dimensions (number of features) = 129 in the project 
  PetscInt m;								// Number of letters in the training data 

  /// Test dataset and its statistics
  char tdata_path[PETSC_MAX_PATH_LEN];  // filename of the test data
  Vec tlabels;							// Test labels
  Mat tdata;								// Test data
  PetscInt tdim;						// test Dimensions (equal to dim)
  PetscInt tm;							// Number of letters in test data  
	
  /// Training parameters
  PetscReal lambda;
	LossName loss;           	// Type of loss that we are running
	PetscReal tol;						// Tolerance 
	
  // Variables for intermediate storage 
  Vec fx;										// Store predictions <w, x> for the training data
  Vec tfx;									// Store predictions for the test data
	Vec c_node;								// Store C_train, the coefficients for the gradient   

	timer opt_timer;					// Timer for the overall optimization procedure 
	timer objgrad_timer;			// Timer for the objective/gradient calculation 
	timer matvec_timer;				// Timer for the X*w cost	

	SeqCtx seq, tseq;					// sequence context for training and test sets
	
	// Below are the variables recommended for CRF loss ONLY
	// If you really would like to try your own ideas, free free to remove.
	Vec 				w_node, w_edge, w_edgeloc;	// edge weights
	Vec 				g_node, g_edge, g_edgeloc;	// gradient of edge weights
	VecScatter  scatter;      // Scatter context	
	Vec 				w_temp;				// A temp vector for computing the gradient of w
	Mat 				M1, M2;				// matrices for extracting the node weights 
														//  and edge weights from the concatenated vector w
	// Below is used for dynamic programming (DP).  
	// If they are removed, you will have to write your own DP code.
	// First three variables are 2-D arrays sized #class * #class (26 * 26)
	// They are used temporarily by DP.
	PetscReal **weight_edge;	// weight_edge[i][j] = T_{ij} is the edge potential/weights
	PetscReal **marg_edge;		// marginal distribution on each edge p(y_s = i, y_{s+1} = j)
	PetscReal **grad_edge;		// gradient of edge weights (just sum up marg_edge over all letters)
														//    and subtract empirical frequency as in Eq 17
	// Next are four 2-D arrays, sized (#letter of the current word) * nclass
	// They are also used temporarily by DP.
	PetscReal **left;					// buffer for dynamic programming
	PetscReal **right;				// buffer for dynamic programming
	PetscReal **score;				// array to store the fx value of the current word
	PetscReal **marg;					// marginal probability of each node p(y_s = k)
	
} AppCtx;


// The file context which is used ONLY in loading the data
// No need to read or digest it (unless you want to delve into how the data is loaded)
typedef struct{
	char					*data_path;
	FILE					*data_file;
	
  PetscInt dim;						// Dimensions
  PetscInt m;							// Number of letters
  PetscInt classes;				// Number of classes (26)
  PetscInt maxnnz;				// Maximum number of non-zero features over all rows/letters
  PetscInt maxlen;				// Maximum number of characters in any row of input file
	SeqCtx	 *seq;					// statistics of the sequences (will fill in some during loading)
	Vec      *labels;				// Store labels (in fact, a pointer to labels or tlabels in AppCtx,
													//		depending on whether we are loading the training or test data)
  Mat      *data;					// Store data (in fact, a pointer to data or tdata in AppCtx,
													//		depending on whether we are loading the training or test data)
  Mat      data_parts; 		// The raw data matrix before it is replicated for #class times

  // nnz_array records how many nonzero features are there in each letter image (row).
	// This array is created during loading and is deallocated upon completion of loading.
	// Its values are used for allocating the memory of the data matrices.
  PetscInt      *nnz_array;
} AppCtx_file;

PetscErrorCode PrintSetting(AppCtx *user);
PetscErrorCode GetMachineInfo(AppCtx *user);
PetscErrorCode make_sparse_matrix(Mat *M, Vec &x, Vec &y, PetscInt col_start);

#endif
