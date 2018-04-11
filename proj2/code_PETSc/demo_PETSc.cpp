#include <petsctao.h>

// This file demonstrates some usages of PETSc 3.7.3

// For simplicity, I just define rank and size as global variables.
PetscErrorCode ierr;
PetscInt rank;	// ID of this process
PetscInt size;	// number of processes (as specified after -n of mpirun)

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

	PetscInt sz_local_x, sz_local_y, sz_x, sz_y;
	PetscInt *diag_nnz, *offdiag_nnz;
	PetscInt cbegin, cend, rbegin, rend;

	// sz_local_x is the local length of x in my process
	ierr = VecGetLocalSize(x, &sz_local_x);	CHKERRQ(ierr);
	ierr = VecGetLocalSize(y, &sz_local_y);	CHKERRQ(ierr);

	// sz_x is the global total length of x
	ierr = VecGetSize(x, &sz_x);	CHKERRQ(ierr); 
	ierr = VecGetSize(y, &sz_y);	CHKERRQ(ierr);	

	if (col_start + sz_y > sz_x)	// all rows must get a single 1
		SETERRQ(PETSC_COMM_SELF, 1, "offset too big!");

	// See detailed explanation of printing functions at:
	// http://www.mcs.anl.gov/petsc/petsc-current/src/sys/examples/tutorials/ex2.c.html
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, 
											"rank = %D, sz_local_x = %D, sz_local_y = %D, sz_y = %D\n",
											rank, sz_local_x, sz_local_y); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); CHKERRQ(ierr);

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
	// See PETSc manual page 61.
	for (PetscInt i = 0; i < sz_local_y; i++)		// enumerate all local rows
	{
		// translate to the global row number
		PetscInt row = rbegin + i;			
		// add col_start to compute the column ID of the element
		PetscInt col = row + col_start;	

		if (col >= cbegin && col < cend)  {		// if the column belongs to the diagonal block
			diag_nnz[i] = 1;
			offdiag_nnz[i] = 0;
		}
		else {		// if the column does not belong to the diagonal block
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
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Let's see the matrix\n"); CHKERRQ(ierr);
	ierr = MatView(*M, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}


// Function to test the implementation in make_sparse_matrix 
PetscErrorCode test_sparse_matrix() {

	Mat M1, M2;
	Vec x, y;
	PetscInt len_x, len_y, col_start;

	PetscFunctionBegin;

	ierr = PetscPrintf(PETSC_COMM_WORLD, "======================\nIn test_sparse_matrix:\n"); CHKERRQ(ierr);
	len_x = 9;
	len_y = 5;

	// x is 9 dimensional, and y is 5 dimensional
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, len_x, &x); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, len_y, &y); CHKERRQ(ierr);
	ierr = VecSet(x, 1); CHKERRQ(ierr);
	ierr = VecSet(y, 1); CHKERRQ(ierr);

	// Create a 5-by-9 sparse matrix
	// 1 0 0 0 0 0 0 0 0
	// 0 1 0 0 0 0 0 0 0
	// 0 0 1 0 0 0 0 0 0
	// 0 0 0 1 0 0 0 0 0
	// 0 0 0 0 1 0 0 0 0
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Offset = 0\n"); CHKERRQ(ierr);
	ierr = make_sparse_matrix(&M1, x, y, 0);	CHKERRQ(ierr);

	// Create a 4-by-9 sparse matrix	
	// 0 0 0 0 0 1 0 0 0
	// 0 0 0 0 0 0 1 0 0
	// 0 0 0 0 0 0 0 1 0
	// 0 0 0 0 0 0 0 0 1
	col_start = len_x - len_y;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Offset = %D\n", col_start); CHKERRQ(ierr);
	ierr = make_sparse_matrix(&M2, x, y, col_start);	CHKERRQ(ierr);

	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&y); CHKERRQ(ierr);
	ierr = MatDestroy(&M1); CHKERRQ(ierr);
	ierr = MatDestroy(&M2); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


// Test the use of scatter 
// copy/gather all entries of a parallel distributed vector to a local vector
// and do it for all processes
PetscErrorCode test_scatter() {

	Vec w;		// The global vector
	Vec wloc;	// The local gathering of all entries of w (not just the local portion of w)
	PetscInt len = 10;
	VecScatter  scatter;      // Scatter context 
	PetscReal *ptr;

	PetscFunctionBegin;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "======================\nIn test_scatter:\n"); CHKERRQ(ierr);

	// First create a 10-dimensional vector w, with all entries being 0
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, len, &w); CHKERRQ(ierr);
	ierr = VecSet(w, 1.0); CHKERRQ(ierr);

	// Scatter w to the local vector "wloc" of ALL processes
	// That is, all processes have the same copy of the whole vector w
	ierr = VecScatterCreateToAll(w, &scatter, &wloc); CHKERRQ(ierr);
	ierr = VecScatterBegin(scatter, w, wloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
	ierr = VecScatterEnd(scatter, w, wloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

	// wloc is a local vector, and so it is easy to assign its values
	// Different processes may set different values (depending on "rank")
	ierr = VecGetArray(wloc, &ptr); CHKERRQ(ierr);
	for (PetscInt i = 0; i < len; i++)
		ptr[i] = rank + i;
	ierr = VecRestoreArray(wloc, &ptr); CHKERRQ(ierr);

	// scatter wloc back to the global w
	// ADD_VALUES means the new value of w[i] will be the sum of 
	//   (1) the original value of w[i]
	//   (2) the sum of the values of wloc[i] over all processes 
	ierr = VecScatterBegin(scatter, wloc, w, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
	ierr = VecScatterEnd(scatter, wloc, w, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);

	// Print the vector w
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Let's see the new vector\n"); CHKERRQ(ierr);
	ierr = VecView(w, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = VecDestroy(&w); CHKERRQ(ierr);
	ierr = VecDestroy(&wloc); CHKERRQ(ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


// Repeat M (a 3-by-5 matrix) for 2 times,
// Then compute M*x and M^T * (M * x)
// x is 10 dimensional, and y is 6 dimensional
PetscErrorCode test_repmat() {

	PetscInt sz_local_x, sz_local_y;
	PetscInt xbegin, xend, ybegin, yend;
	Mat M_base, M;
	Vec x, y, z;
	PetscReal *ptr;

	PetscFunctionBegin;
	ierr = PetscPrintf(PETSC_COMM_WORLD, "======================\nIn test_repmat:\n"); CHKERRQ(ierr);

	PetscInt len_x = 5;
	PetscInt len_y = 3;
	PetscInt numRep = 2;

	// First create x and y as 5 and 3 dimensional vectors, based on which we construct M_base
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, len_x, &x); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, len_y, &y); CHKERRQ(ierr);
	ierr = VecSet(x, 1); CHKERRQ(ierr);
	ierr = VecSet(y, 1); CHKERRQ(ierr);

	// Now M_base is
	// 1 0 0 0 0
	// 0 1 0 0 0
	// 0 0 1 0 0
	ierr = make_sparse_matrix(&M_base, x, y, 0);	CHKERRQ(ierr);
	// Replicate M for 2 times
	ierr = MatCreateMAIJ(M_base, numRep, &M); CHKERRQ(ierr);  

// I commented out printing of the replicated matrix because it is not shown correctly.
// MatCreateMAIJ constructs a new matrix only "logically", and so printing is not meaningful.
// 	ierr = PetscPrintf(PETSC_COMM_WORLD, "Let's see the repeated matrix\n"); CHKERRQ(ierr);
// 	ierr = MatView(M, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "=====================\n"); CHKERRQ(ierr);

	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&y); CHKERRQ(ierr);

	// Re-create x and y to be consistent with the local #row/#column of M
	ierr = MatGetLocalSize(M, &sz_local_y, &sz_local_x); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, sz_local_x, PETSC_DETERMINE, &x); CHKERRQ(ierr);
	ierr = VecCreateMPI(PETSC_COMM_WORLD, sz_local_y, PETSC_DETERMINE, &y); CHKERRQ(ierr);
	ierr = VecDuplicate(x, &z); CHKERRQ(ierr);

	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,
																	"rank = %D, sz_local_x = %D, sz_local_y = %D\n",
																	rank, sz_local_x, sz_local_y); CHKERRQ(ierr);
	ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); CHKERRQ(ierr);

	// Get the index range of the local rows and columns of M 
	//	(the index values are in the global sense)
	// Recall the local size of x and y are consistent with the local #column/#row of M
	// The two functions below can be called only if the matrix space has been allocated
	//   i.e. MatXXXSetPreallocation has been run.
	ierr = MatGetOwnershipRange(M, &ybegin, &yend); CHKERRQ(ierr);
	ierr = MatGetOwnershipRangeColumn(M, &xbegin, &xend); CHKERRQ(ierr);
	
	// Set the value of x to (0, 1, 2, ..., 9)^T
	ierr = VecGetArray(x, &ptr); CHKERRQ(ierr);
	for (PetscInt i = 0; i < sz_local_x; i++)		// traverse the local elements of x
		ptr[i] = xbegin + i;		// xbegin + i is the global row ID in x
	ierr = VecRestoreArray(x, &ptr); CHKERRQ(ierr);

	ierr = PetscPrintf(PETSC_COMM_WORLD, "Vector x = \n"); CHKERRQ(ierr);
	ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Compute y = M*x.  The resulting y is (0, 1, 2, 3, 4, 5)^T
	ierr = MatMult(M, x, y);  CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Vector y = \n"); CHKERRQ(ierr);
	ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	// Compute z = M^T * y.  The resulting z is (0, 1, 2, 3, 4, 5, 0, 0, 0, 0)^T
	ierr = MatMultTranspose(M, y, z); CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD, "Vector z = \n"); CHKERRQ(ierr);
	ierr = VecView(z, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = VecDestroy(&y); CHKERRQ(ierr);
	ierr = VecDestroy(&z); CHKERRQ(ierr);
	ierr = MatDestroy(&M_base); CHKERRQ(ierr);
	ierr = MatDestroy(&M); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


PetscErrorCode main(int argc, char **argv) {

	// Initialize TAO and PETSc
	PetscInitialize(&argc, &argv, (char *)0, NULL);

	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	ierr = test_sparse_matrix(); CHKERRQ(ierr);
	ierr = test_scatter(); CHKERRQ(ierr);
	ierr = test_repmat(); CHKERRQ(ierr);

	ierr = PetscFinalize(); CHKERRQ(ierr);
	return 0;
}