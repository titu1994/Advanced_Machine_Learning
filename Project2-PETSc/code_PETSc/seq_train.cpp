#include <petsctao.h>
#include "appctx.hpp"
#include "loaddata_libsvm.hpp"
#include "iid_loss.hpp"
#include "crf_loss.hpp"

static  char help[] = "sequence prediction using TAO \n\
Input parameters are:\n\
  -data    <filename> : file which contains training data\n\
  -tdata   <filename> : file which contains test data\n\
	-loss    <string>   : (must specify) CRF or IID\n \
  -lambda <double> : the regularization parameter (def: 1e-3) \n\
  -tol <double> : convergence tolerance (def: 1e-5) \n\n";

// Actual loss and gradient computation happens in this function     
PetscErrorCode(*LossGrad)(Tao tao, Vec w, double *f, Vec G, void *ctx);

// Compute performance on the training/test dataset
PetscErrorCode(*Evaluate)(Vec w, AppCtx* user);

// Allocate the space for work
PetscErrorCode(*AllocateWorkSpace)(Vec *w, AppCtx *user);

// Destroy the work space
PetscErrorCode(*DestroyWorkSpace)(Vec *w, AppCtx *user);

// Show in each iteration the objective value and 
// training/test performance of intermediate solutions 
PetscErrorCode Monitor(Tao tao, void *ctx);

// Parse the arguments in the command line
PetscErrorCode LoadOptions(AppCtx *user);


int main(int argc, char **argv) {
  PetscErrorCode       ierr;  // used to check for functions returning nonzeros
  Tao                  tao;                             
  Vec                  w;			// the optimization variable
  AppCtx               user;  // context of the optimization problem (e.g. data)   
  
  // Initialize TAO and PETSc
  PetscInitialize(&argc, &argv, (char *)0, help);
  
  ierr = GetMachineInfo(&user);  CHKERRQ(ierr);
  
  // Check for command line arguments to override defaults
	ierr = LoadOptions(&user); CHKERRQ(ierr);
	
	// Load the training/test data from data files
	ierr = LoadDataLibsvm(&user); CHKERRQ(ierr);
	
	// Allocate variables for working
	ierr = AllocateWorkSpace(&w, &user); CHKERRQ(ierr);   // Allocate space 
	
	if (user.rank == 0)
    PrintSetting(&user);

  // Initialize the solution to the zero vector
	ierr = VecSet(w, 0.0); CHKERRQ(ierr);

	// Create the TAO solver with desired solution method 
	ierr = TaoCreate(PETSC_COMM_WORLD, &tao); CHKERRQ(ierr);
	
  // Set the TAO solver to BFGS
	ierr = TaoSetType(tao, TAOLMVM); CHKERRQ(ierr);

  // Attach w to the TAO solver as the solution vector (to be optimized over)
	ierr = TaoSetInitialVector(tao, w); CHKERRQ(ierr);
	
	// Set the routine of computing the objective value and gradient
	ierr = TaoSetObjectiveAndGradientRoutine(tao, LossGrad, &user); CHKERRQ(ierr);

	// Set convergence tolerance.  See its meanings at
	// http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Tao/TaoSetTolerances.html
	ierr = TaoSetTolerances(tao, user.tol, 0, user.tol); CHKERRQ(ierr);	
	// Set the max number of BFGS iteration
	ierr = TaoSetMaximumIterations(tao, 1000); CHKERRQ(ierr);		
	// Set the max number of function evaluation
	ierr = TaoSetMaximumFunctionEvaluations(tao, 3000); CHKERRQ(ierr);	

	// Print intermediate function values and training/test performance
	ierr = TaoSetMonitor(tao, Monitor, &user, PETSC_NULL); CHKERRQ(ierr);
	
	// Check for TAO command line options 
	ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);
  
	// Solve the optimization problem 
	if (user.rank == 0) 
		PetscPrintf(PETSC_COMM_SELF, "iter fn.val gap time feval.num train_lett_err train_word_err test_lett_err test_word_err \n");
  user.opt_timer.start();		// Start the timer 
	ierr = TaoSolve(tao); CHKERRQ(ierr);  
  user.opt_timer.stop();	// Stop the timer
  
  // Show the reason why TAO terminated. Code of reason available at 
  // https://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/Tao/TaoConvergedReason.html
  TaoConvergedReason reason;  
  ierr = TaoGetConvergedReason(tao, &reason); CHKERRQ(ierr);  
	if (reason < 0) 
		PetscPrintf(MPI_COMM_WORLD, "Optimization terminated due to %s.\n", TaoConvergedReasons[reason]);
	else
		PetscPrintf(MPI_COMM_WORLD, "Optimization converged with status %s.\n", TaoConvergedReasons[reason]);

  // Print statistics on timing 
  if (user.rank == 0) {
    PetscPrintf(PETSC_COMM_WORLD, "Total CPU time: %g seconds \n", user.opt_timer.cpu_total);
    PetscPrintf(PETSC_COMM_WORLD, "Total wallclock time: %g seconds \n", user.opt_timer.wallclock_total);
    PetscPrintf(PETSC_COMM_WORLD, "Objective/gradient CPU time: %g seconds \n", user.objgrad_timer.cpu_total);
    PetscPrintf(PETSC_COMM_WORLD, "Objective/gradient wallclock time: %g seconds \n", user.objgrad_timer.wallclock_total);    
  }
 
  // Free all work space
  DestroyWorkSpace(&w, &user);
    
  // Finalize TAO and PETSc
  ierr = PetscFinalize(); CHKERRQ(ierr);  
  return 0;
}


PetscErrorCode LoadOptions(AppCtx *user) {
  PetscBool         flg;
  PetscErrorCode    ierr;

	PetscFunctionBegin;	
	user->lambda = 1e-3;		// default values
  user->tol  = 1e-5;

	// Read the file name of training data from the command line
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-data", user->data_path, PETSC_MAX_PATH_LEN - 1, &flg); CHKERRQ(ierr);
  if (flg == PETSC_FALSE)
    SETERRQ(PETSC_COMM_WORLD, 1, "No train files specified!");
  
  // Read the file name of test data from the command line  
  ierr = PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-tdata", user->tdata_path, PETSC_MAX_PATH_LEN - 1, &flg); CHKERRQ(ierr);
  if (flg == PETSC_FALSE)
    SETERRQ(PETSC_COMM_WORLD, 1, "No test files specified!");
  
  // Read the value of \lambda from the command line
	ierr = PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-lambda", &user->lambda, &flg); CHKERRQ(ierr);
	
	// Read the optimization tolerance from the command line
  ierr = PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-tol", &user->tol, &flg); CHKERRQ(ierr);
	  
	// Read the loss specification from the command line (IID or CRF)
	ierr = PetscOptionsGetEnum(PETSC_NULL, PETSC_NULL, "-loss", LossNames,
															(PetscEnum*)&(user->loss), &flg); CHKERRQ(ierr);
	if (flg == PETSC_FALSE)
		SETERRQ(PETSC_COMM_WORLD, 1, "loss is not set!");

	switch (user->loss){
	case LOSS_IID:
		PetscPrintf(PETSC_COMM_WORLD, "IID loss\n");
		LossGrad = iid_loss::LossGrad;
		Evaluate = iid_loss::Evaluate;
		AllocateWorkSpace = iid_loss::AllocateWorkSpace;
		DestroyWorkSpace = iid_loss::DestroyWorkSpace;
		break;
	case LOSS_CRF:
		PetscPrintf(PETSC_COMM_WORLD, "CRF loss\n");
		LossGrad = crf_loss::LossGrad;
		Evaluate = crf_loss::Evaluate;
		AllocateWorkSpace = crf_loss::AllocateWorkSpace;
		DestroyWorkSpace = crf_loss::DestroyWorkSpace;
		break;
	}
	PetscFunctionReturn(0);
}


// Monitor - Print intermediate function values and training/test performance
// Input Parameters:
// 	tao     - tao solver
// 	ctx     - user defined application context 
// Output Parameters:
// 	None. Just display intermediate function values and training/test performance
PetscErrorCode Monitor(Tao tao, void *ctx) {
  AppCtx *user = (AppCtx *) ctx;
  PetscErrorCode ierr;		
	PetscScalar f, gnorm;		// current objective value, and gradient norm
  PetscInt iter;					// how many iterations TAO has run
  Vec w;

	PetscFunctionBegin;
	user->opt_timer.stop();	// pause the timer (time for monitoring is not the training cost)
  
	// Get the current number of iteration (iter), 
	//    objective function value (f), and gradient norm (gnorm)
  ierr = TaoGetSolutionStatus(tao, &iter, &f, &gnorm, PETSC_NULL, PETSC_NULL, PETSC_NULL);	CHKERRQ(ierr);
  
  PetscInt nfuncs;		// See how many times the objective function has been evaluated
  ierr = TaoGetCurrentFunctionEvaluations(tao, &nfuncs);	CHKERRQ(ierr);
  
  if (user->rank == 0)
    PetscPrintf(PETSC_COMM_SELF, "%D %g %g %g %D ", iter, f, gnorm, user->opt_timer.wallclock_total, nfuncs);
  
  ierr = TaoGetSolutionVector(tao, &w); CHKERRQ(ierr);	// retrieve the current solution w
	ierr = Evaluate(w, user); CHKERRQ(ierr);	// print performance measure on training/test data
  user->opt_timer.start();	// resume the timer
	PetscFunctionReturn(0);
}
