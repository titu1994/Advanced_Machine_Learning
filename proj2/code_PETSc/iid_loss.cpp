#include "iid_loss.hpp"
#include <algorithm>

namespace iid_loss{
  
  // loss_coef - Evaluates loss and its gradient for IID model
  // Input Parameters:
  //	 fx: function prediction
  //	 labels: labels	 
  //	 user: user defined application context 
	//	 seq: the sequence information context

  // Output variables:
  //	 c_node: the c_{local} vector, encoding the coefficients for computing the gradient   
  //	 loss: this process' contribution to function value
  PetscErrorCode loss_coef(Vec &fx, 
                           Vec &labels, 
                           Vec &c_node,
                           PetscReal *loss,
                           AppCtx      *user,
                           SeqCtx			 *seq) {
    PetscErrorCode ierr;
    PetscReal *fx_array, *labels_array, *cnode_array;
    PetscFunctionBegin;
    *loss = 0.0;
    
    ierr = VecGetArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecGetArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecGetArray(c_node, &cnode_array);  CHKERRQ(ierr);
    
    // Compute function and gradient coefficients
    // Enumerate all local *letters*
		for (PetscInt i = 0; i < seq->lLocalCount; i++) {
      PetscInt start = i * user->nclasses;
      PetscInt end = (i+1) * user->nclasses;
			PetscReal fx_max = *std::max_element(fx_array + start, fx_array + end);
      PetscReal fx_sum = 0.0;
      for(PetscInt idx = start; idx < end; idx++) {
				cnode_array[idx] = exp(fx_array[idx] - fx_max);
        fx_sum += cnode_array[idx];
      }
      PetscInt lidx = labels_array[i] + start;
      *loss -= (fx_array[lidx] - log(fx_sum) - fx_max);
			for (PetscInt idx = start; idx < end; idx++)
				cnode_array[idx] /= fx_sum * seq->wGlobalCount;
			cnode_array[lidx] -= 1.0 / seq->wGlobalCount;			
    }
    *loss /= seq->wGlobalCount;
    
    ierr = VecRestoreArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(c_node, &cnode_array);  CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  
	// LossGrad - Evaluates a scalar loss and its gradient. 
	// Input Parameters:
	// 	 w      - current weight vector
	// 	 ctx    - optional user-defined context
	// Output Parameters:
	//   G 			- vector containing the newly evaluated gradient
	//   f 			- function value
	PetscErrorCode LossGrad(Tao tao, Vec w, double *f, Vec G, void *ctx) {
		AppCtx *user = (AppCtx *)ctx;
		PetscErrorCode ierr;
		PetscReal reg = 0.0;
		
		PetscFunctionBegin;

  	user->objgrad_timer.start();

		user->matvec_timer.start();			
		ierr = MatMult(user->data, w, user->fx); CHKERRQ(ierr);
		user->matvec_timer.stop();

		// Computes the function and gradient coefficients 
		ierr = loss_coef(user->fx, user->labels, user->c_node, f, user, &user->seq);	CHKERRQ(ierr);
		
		// Sum up the contribution of loss from all processes
		MPI_Allreduce(MPI_IN_PLACE, f, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

		// Compute the regularization
		ierr = VecDot(w, w, &reg); CHKERRQ(ierr);
		*f = *f + reg * user->lambda / 2.0;

		// Now everyone can compute the gradient
		user->matvec_timer.start();
		ierr = MatMultTranspose(user->data, user->c_node, G); CHKERRQ(ierr);
		user->matvec_timer.stop();

		ierr = VecAXPY(G, user->lambda, w); CHKERRQ(ierr);

		user->objgrad_timer.stop();
		PetscFunctionReturn(0);
	}

	// Get the letter-wise and word-wise error of for the local sub-dataset
	// Input Parameters:
  // 	 fx: function prediction (X*w)
  // 	 labels: labels
  // 	 user: user defined application context 
  // 	 seq: sequence information context
  // Output parameters:
  // 	 lError: the letter-wise error rate
  // 	 wError: the word-wise error rate
	PetscErrorCode get_errors(Vec &fx, Vec &labels, SeqCtx *seq, 
													  AppCtx* user, PetscInt *lError, PetscInt *wError)	{
		PetscErrorCode    ierr;
		PetscReal *fx_array, *labels_array;
		PetscInt nClass = user->nclasses;

		PetscFunctionBegin;
		ierr = VecGetArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecGetArray(labels, &labels_array); CHKERRQ(ierr);

		*lError = *wError = 0;
		for (PetscInt i = seq->wBegin; i < seq->wEnd; i++)		{
			PetscBool wFail = PETSC_FALSE;
			for (PetscInt j = 0; j < seq->wLen[i]; j++) {
				PetscInt maxInd = std::distance(fx_array, std::max_element(fx_array, fx_array + nClass));
				if (maxInd != (int)(*labels_array+0.5))  {
					(*lError)++;
					wFail = PETSC_TRUE;
				}
				fx_array += nClass;		labels_array++;
			}
			if (wFail)  (*wError)++;
		}

		fx_array -= nClass * seq->lLocalCount;
		labels_array -= seq->lLocalCount;
		ierr = VecRestoreArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(labels, &labels_array); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}

	// Evaluate - Evaluate letter-wise and word-wise error rates on training and test set
	// Input Parameters:
	// 	 w       - current weight vector
	// 	 user    - Application context
	// Output Parameters:
	// 	 None. Display the letter-wise and word-wise error rates 
	PetscErrorCode Evaluate(Vec w, AppCtx* user) {
		PetscErrorCode    ierr;
		PetscInt lError, wError;

		PetscFunctionBegin;
		
		// Compute the model output discriminative values on training data
		ierr = MatMult(user->data, w, user->fx); CHKERRQ(ierr);

		// Get the error of word and letter for the local sub-dataset of training data
		ierr = get_errors(user->fx, user->labels, &user->seq, user, &lError, &wError); CHKERRQ(ierr);

		// Sum up the errors (both word wise and letter wise)
		MPI_Allreduce(MPI_IN_PLACE, &lError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &wError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

		if (user->rank == 0)
			PetscPrintf(PETSC_COMM_SELF, "%f %f ", 
									lError*100.0/user->m, wError*100.0/user->seq.wGlobalCount);

		// Repeat the above for test data
		// Compute the model output discriminative values on test data
		ierr = MatMult(user->tdata, w, user->tfx); CHKERRQ(ierr);

		// Get the error of word and letter for the local sub-dataset of test data
		ierr = get_errors(user->tfx, user->tlabels, &user->tseq, user, &lError, &wError); CHKERRQ(ierr);

		// Sum up the errors (both word wise and letter wise)
		MPI_Allreduce(MPI_IN_PLACE, &lError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &wError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

		if (user->rank == 0)
			PetscPrintf(PETSC_COMM_SELF, "%f %f\n", 
									lError*100.0 / user->tm, wError*100.0 / user->tseq.wGlobalCount);
		PetscFunctionReturn(0);
	}

	// Allocate the work variables
	PetscErrorCode AllocateWorkSpace(Vec *w, AppCtx *user) {
		PetscErrorCode     ierr;
		PetscInt  m_local, tm_local, dim_local;

		PetscFunctionBegin;
		ierr = MatGetLocalSize(user->data, &m_local, &dim_local); CHKERRQ(ierr);
		ierr = MatGetLocalSize(user->tdata, &tm_local, &dim_local); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, dim_local, PETSC_DETERMINE, w); CHKERRQ(ierr);
		ierr = VecSetFromOptions(*w); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, m_local, PETSC_DETERMINE, &user->fx); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->fx); CHKERRQ(ierr);
		ierr = VecDuplicate(user->fx, &user->c_node); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, tm_local, PETSC_DETERMINE, &user->tfx); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->tfx); CHKERRQ(ierr);

		PetscFunctionReturn(0);
	}

	// Destroy the work variables
	PetscErrorCode DestroyWorkSpace(Vec *w, AppCtx *user) {		
		PetscErrorCode     ierr;
		PetscFunctionBegin;
		
		// Free some space
		ierr = MatDestroy(&user->data); CHKERRQ(ierr);
		ierr = VecDestroy(&user->labels); CHKERRQ(ierr);
		ierr = VecDestroy(&user->fx); CHKERRQ(ierr);

		ierr = MatDestroy(&user->tdata); CHKERRQ(ierr);
		ierr = VecDestroy(&user->tlabels); CHKERRQ(ierr);
		ierr = VecDestroy(&user->tfx); CHKERRQ(ierr);

		ierr = VecDestroy(w); CHKERRQ(ierr);
		ierr = VecDestroy(&user->c_node); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}

}
