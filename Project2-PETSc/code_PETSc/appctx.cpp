#include "appctx.hpp"

const char *LossNames[] = { "IID", "CRF", "LossName", "LOSS_", 0 };

PetscErrorCode PrintSetting(AppCtx *user) {
  PetscErrorCode ierr;
  
	PetscFunctionBegin;
	PetscPrintf(PETSC_COMM_WORLD, "====================  Summary of settings =================\n");

	ierr = PetscPrintf(PETSC_COMM_SELF, "Training data: %s\n", user->data_path); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_SELF, "Test data: %s\n", user->tdata_path); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "loss: %s, ", LossNames[user->loss]); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "lambda: %e, ", user->lambda); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_SELF, "Processors: %D\n", user->size); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "dim: %D, ", user->dim);
  ierr = PetscPrintf(PETSC_COMM_SELF, "classes: %D\n", user->nclasses);
	ierr = PetscPrintf(PETSC_COMM_SELF, "Train: #word=%D, #letter=%D\n", user->seq.wGlobalCount, user->m); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_SELF, "Test: #word=%D, #letter=%D\n", user->tseq.wGlobalCount, user->tm); CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "===========================================================\n");
	PetscFunctionReturn(0);
}


// Print the name of the machines that the code is running on
PetscErrorCode GetMachineInfo(AppCtx *user)
{
  char hostname[1024];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &user->rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &user->size);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%d machines found\n", user->size);  CHKERRQ(ierr);
  gethostname(hostname, 1024);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s\n", hostname);  CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


// Identical to the same-named routine in demo_PETSc.cpp
// Make a sparse matrix M sized length(y)-by-length(x) as
// for (i = 0; i < length(y); i ++)
//     M(i, i + col_start) = 1
// end
// All other entries of M are 0.
// Hardness: make the local #row of M equal local_length(y)
//                and the local #column (diagonal) of M equal local_length(x)
//           This will allow y = M * x to be called.
PetscErrorCode make_sparse_matrix(Mat *M, Vec &x, Vec &y, PetscInt col_start) {
	PetscFunctionBegin;

	PetscErrorCode ierr;
	PetscInt sz_local_x, sz_local_y, sz_x, sz_y;
	PetscInt *diag_nnz, *offdiag_nnz;
	PetscInt cbegin, cend, rbegin, rend;
	PetscInt rank;

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	// sz_local_x is the local length of x in my process
	ierr = VecGetLocalSize(x, &sz_local_x);	CHKERRQ(ierr);
	ierr = VecGetLocalSize(y, &sz_local_y);	CHKERRQ(ierr);

	// sz_x is the global total length of x
	ierr = VecGetSize(x, &sz_x);	CHKERRQ(ierr);
	ierr = VecGetSize(y, &sz_y);	CHKERRQ(ierr);

	if (col_start + sz_y > sz_x)	// all rows must get a single 1
		SETERRQ(PETSC_COMM_SELF, 1, "offset too big!");

	ierr = MatCreate(PETSC_COMM_WORLD, M); CHKERRQ(ierr);
	// Create a matrix, with local #row = sz_local_y, and local #column = sz_local_x
	// The global total #row/#column is left to PETSc to determine (just sum them up)
	ierr = MatSetSizes(*M, sz_local_y, sz_local_x, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*M, MATMPIAIJ); CHKERRQ(ierr);

	ierr = PetscMalloc(sz_local_y*sizeof(PetscInt), &diag_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(sz_local_y*sizeof(PetscInt), &offdiag_nnz); CHKERRQ(ierr);

	// Now we compute the local row and column index range of M 
	//	(the index values are in the global sense).
	// This is achieved by using the number of local rows/columns in each process.
	// MPI_Exscan computes the partial sum of sz_local_x over all processes
	// For process "rank", cbegin is the sum of sz_local_x of processes 0, ..., rank-1	
	MPI_Exscan(&sz_local_x, &cbegin, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
	MPI_Exscan(&sz_local_y, &rbegin, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
	if (rank == 0)  {	// partial sum is undefined for process 0, so set it manually
		cbegin = rbegin = 0;
	}
	cend = cbegin + sz_local_x;
	rend = rbegin + sz_local_y;

	// Now for each row, set up the number of nonzero elements in the (non) diagonal block 
	for (PetscInt i = 0; i < sz_local_y; i++)		// enumerate all local rows
	{
		// translate to the global row number
		PetscInt row = rbegin + i;
		// add col_start to compute the column ID of the element
		PetscInt col = row + col_start;

		if (col >= cbegin && col < cend)  {		// if belong to the diagonal block
			diag_nnz[i] = 1;
			offdiag_nnz[i] = 0;
		}
		else {		// if not belong to the diagonal block
			diag_nnz[i] = 0;
			offdiag_nnz[i] = 1;
		}
	}

	// Allocate memory for the sparse matrix
	ierr = MatMPIAIJSetPreallocation(*M, 0, diag_nnz, 0, offdiag_nnz); CHKERRQ(ierr);

	// put 1 at all nonzero entries
	for (PetscInt i = 0; i < sz_local_y; i++)	{
		PetscInt row = rbegin + i;
		PetscInt col = row + col_start;
		PetscReal val = 1.0;
		ierr = MatSetValues(*M, 1, &row, 1, &col, &val, INSERT_VALUES); CHKERRQ(ierr);
	}

	// assemble the entry values
	ierr = MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = PetscFree(diag_nnz); CHKERRQ(ierr);	// delete auxilliary arrays
	ierr = PetscFree(offdiag_nnz); CHKERRQ(ierr);

	// Print the matrix
// 	ierr = PetscPrintf(PETSC_COMM_WORLD, "Let's see the matrix\n"); CHKERRQ(ierr);
// 	ierr = MatView(*M, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}
