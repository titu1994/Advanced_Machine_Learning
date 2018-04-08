#include "crf_loss.hpp"
#include <algorithm>
#include <numeric>

namespace crf_loss{

	// loss_grad - Evaluates logistic loss and its gradient.     
	// Input Parameters:
	// 	 fx: 				function prediction (X*w)
	// 	 labels: 		labels
	// 	 w_edgeloc: edge weights   
	// 	 user: 			user defined application context 
	// 	 seq: 			sequence information context

	// Output parameters:
	// 	 c_node: 		the c_{local} vector, encoding the coefficients for computing the gradient
	// 	 g_edgeloc: gradient of edge weights
	// 	 loss: 			this process' contribution to function value
	PetscErrorCode loss_coef(Vec 	&fx, 
			Vec 	&labels,      
			Vec 	&w_edgeloc,                      
			Vec 	&c_node, 
			Vec 	&g_edgeloc,
			PetscReal 	*loss,
			AppCtx     *user,
			SeqCtx			*seq)  {
		PetscErrorCode ierr;
		PetscInt i, j, k, letter_sofar, nClass, nextLabel, curLabel;
		PetscReal *label, maxTemp, denom;
		PetscReal *buf;
		// Arrays to access the input and output vectors
		PetscReal *fx_array, *labels_array, *cnode_array;
		PetscReal *w_edge_array, *g_edge_array;		
		// Give some shorter names to the persistent work variables in AppCtx
		PetscReal **left = user->left, **right = user->right;
		PetscReal **score = user->score, **marg = user->marg;
		PetscReal **weight_edge = user->weight_edge, **marg_edge = user->marg_edge, **grad_edge = user->grad_edge;

		PetscFunctionBegin;
		*loss = 0.0;
		letter_sofar = 0;		// So far, 0 letters have been processed
		nClass = user->nclasses;

		ierr = VecGetArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecGetArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecGetArray(c_node, &cnode_array);  CHKERRQ(ierr);
		ierr = VecGetArray(w_edgeloc, &w_edge_array); CHKERRQ(ierr);
		ierr = VecGetArray(g_edgeloc, &g_edge_array); CHKERRQ(ierr);		

		ierr = PetscMemzero(g_edge_array, nClass*nClass*sizeof(PetscReal));  CHKERRQ(ierr);
		ierr = PetscMalloc(user->nclasses * sizeof(PetscReal), &buf); CHKERRQ(ierr);

		// weight_edge[i][j] = T_{ij}
		// turn the w_edgeloc/g_edgeloc vector into a 2D array form
		weight_edge[0] = w_edge_array;		
		grad_edge[0] = g_edge_array;		
		for (j=1; j < nClass; j++)  {
			weight_edge[j] = weight_edge[j-1] + nClass;
			grad_edge[j] = grad_edge[j-1] + nClass;
		}

		// Compute function and gradient coefficients
		// loop through words, then each character in the word
		for (PetscInt w = seq->wBegin; w < seq->wEnd; w++) {
			PetscInt wLen = seq->wLen[w];

			// score: array to store the fx value of the current word
			score[0] = fx_array + letter_sofar*nClass;
			// marg: marginal probability of each node p(y_s = k)
			marg[0] = cnode_array + letter_sofar*nClass;
			label = labels_array + letter_sofar;
			letter_sofar += wLen;			

			// still, 2d array form
			// score[i] corresponds the array of i-th letter to all the classes, same with marg
			for (i = 1; i < wLen; i++)	{
				// score[i][j]: score (linear discriminant value) for i-th letter and j-th class
				score[i] = score[i - 1] + nClass;
				marg[i] = marg[i - 1] + nClass;
			}

			// Compute the left (forward) messages
			// starts with zero
			for (j = 0; j < nClass; j++)  
				left[0][j] = right[wLen-1][j] = 0.0;
			for (i = 1; i < wLen; i++)  {
				for (j = 0; j < nClass; j++)  {		  		
					for (k = 0; k < nClass; k++)  { 
						// k: index for the class of the preceding letter
						// buf size: 26
						buf[k] = left[i-1][k] + score[i-1][k] + weight_edge[k][j];
					}

					maxTemp = *std::max_element(buf, buf+nClass);
					left[i][j] = 0;
					for (k=0; k < nClass; k++)
						left[i][j] += exp(buf[k] - maxTemp);
					// log(sum(exp(buf[k])))
					left[i][j] = log(left[i][j]) + maxTemp;
				}
			}		  

			// Compute the right (backward) messages		  
			for (i = wLen-2; i >= 0; i--)  {
				for (j = 0; j < nClass; j++)  {
					for (k = 0; k < nClass; k++)  { 
						// k: index for the class of the next letter
						buf[k] = right[i+1][k] + score[i+1][k] + weight_edge[j][k];		  			
					}

					maxTemp = *std::max_element(buf, buf+nClass);
					right[i][j] = 0;
					for (k=0; k < nClass; k++)
						right[i][j] += exp(buf[k] - maxTemp);
					right[i][j] = log(right[i][j]) + maxTemp;
				}
			}		  

			// Now compute the gradient on node weights
			for (i=0; i < wLen; i++)  {
				curLabel = (int)(label[i]+0.5);		  		

				for (j = 0; j < nClass; j++)  {
					left[i][j] += score[i][j];
					right[i][j] += score[i][j];
					buf[j] = left[i][j] + right[i][j] - score[i][j];		  		
				}
				maxTemp = *std::max_element(buf, buf+nClass);

				for (j = 0; j < nClass; j++)
					buf[j] = exp(buf[j] - maxTemp);
				denom = std::accumulate(buf, buf+nClass, 0.0);

				if (i == 0)
					*loss += log(denom) + maxTemp;
				*loss -= score[i][curLabel];

				// Set the marginal probability of each letter
				denom *= seq->wGlobalCount;
				for (j = 0; j < nClass; j++)  
					marg[i][j] = buf[j] / denom;		  		
				marg[i][curLabel] -= 1.0/seq->wGlobalCount;
			}

			// Now compute the gradient of edge weights 
			for (i=0; i < wLen - 1; i++)  {		  	
				curLabel = (int)(label[i]+0.5);
				nextLabel = (int)(label[i+1]+0.5);
				*loss -= weight_edge[curLabel][nextLabel];

				for (j = 0; j < nClass; j++)  
					for (k = 0; k < nClass; k++)
						marg_edge[j][k] = weight_edge[j][k] + left[i][j] + right[i+1][k];
				maxTemp = *std::max_element(marg_edge[0], marg_edge[0]+nClass*nClass);		  				  

				for (j = 0; j < nClass; j++)  
					for (k = 0; k < nClass; k++)
						marg_edge[j][k] = exp(marg_edge[j][k] - maxTemp);
				denom = std::accumulate(marg_edge[0], marg_edge[0]+nClass*nClass, 0.0);

				denom *= seq->wGlobalCount;
				for (j = 0; j < nClass; j++)  
					for (k = 0; k < nClass; k++)
						grad_edge[j][k] += marg_edge[j][k] / denom;
				grad_edge[curLabel][nextLabel] -= 1.0 / seq->wGlobalCount;
			}
		}
		*loss /= seq->wGlobalCount;
		ierr = PetscFree(buf); CHKERRQ(ierr);		

		ierr = VecRestoreArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(c_node, &cnode_array);  CHKERRQ(ierr);
		ierr = VecRestoreArray(w_edgeloc, &w_edge_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(g_edgeloc, &g_edge_array); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}

	PetscErrorCode Scatter(AppCtx *user, int op) {
		PetscErrorCode ierr;
		PetscFunctionBegin;
		if (op == 0) {
			ierr = VecScatterBegin(user->scatter, user->w_edge, user->w_edgeloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
			ierr = VecScatterEnd(user->scatter, user->w_edge, user->w_edgeloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
		} else {
			ierr = VecScatterBegin(user->scatter, user->g_edgeloc, user->g_edge, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
			ierr = VecScatterEnd(user->scatter, user->g_edgeloc, user->g_edge, ADD_VALUES, SCATTER_REVERSE); CHKERRQ(ierr);
		}
		PetscFunctionReturn(0);
	}

	// LossGrad - Evaluates the objective of CRF training and its gradient. 
	// Input Parameters:
	// 	 w       - current weight vector
	// 	 ctx     - optional user-defined context
	// Output Parameters:
	// 	 G 			- vector containing the newly evaluated gradient
	// 	 f 			- function value
	PetscErrorCode LossGrad(Tao tao, Vec w, double *f, Vec G, void *ctx) {
		AppCtx *user = (AppCtx *)ctx;
		PetscErrorCode ierr;
		PetscReal node_regularization = 0.0;
		PetscReal edge_regularization = 0.0;

		PetscFunctionBegin;

		user->objgrad_timer.start();
		user->matvec_timer.start();	

		// get w_node and w_edge first
		ierr = MatMult(user->M1, w, user->w_node); CHKERRQ(ierr);	
		ierr = MatMult(user->M2, w, user->w_edge); CHKERRQ(ierr);
		ierr = MatMult(user->data, user->w_node, user->fx); CHKERRQ(ierr);
		user->matvec_timer.stop();

		// scatter w_edge to each worker
		ierr = Scatter(user, 0); CHKERRQ(ierr);

		// compute fn and and Ctrain for grad calc 
		ierr = loss_coef(user->fx, user->labels, user->w_edgeloc, user->c_node, user->g_edgeloc, f, user, &user->seq);	CHKERRQ(ierr);

		// Sum up losses from each worker
		MPI_Allreduce(MPI_IN_PLACE, f, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);	
			
		// Compute sum(W^2) node regularization
		ierr = VecDot(user->w_node, user->w_node, &node_regularization); CHKERRQ(ierr);
		// Compute sum(Tij^2) edge regularization
		ierr = VecDot(user->w_edge, user->w_edge, &edge_regularization); CHKERRQ(ierr);
		*f = *f + (node_regularization + edge_regularization / 2.0) * user->lambda / 2.0;

		// Pass g_edgeloc to all workers
		ierr = VecSet(user->g_edge, 0.0); CHKERRQ(ierr);
		ierr = Scatter(user, 1); CHKERRQ(ierr);

		// Compute gradient X_train^T * C_train
		user->matvec_timer.start();
		ierr = MatMultTranspose(user->data, user->c_node, user->g_node); CHKERRQ(ierr);
		user->matvec_timer.stop();

		// Compute w_node = lambda * g_node + w_node
		ierr = VecAXPY(user->g_node, user->lambda, user->w_node); CHKERRQ(ierr);
		// Compute w_edge = lambda/2.0 * g_edge + w_edge
		ierr = VecAXPY(user->g_edge, user->lambda/2.0, user->w_edge); CHKERRQ(ierr);

		// Now, we need to concatenate back g_node and g_edge using M1&M2 which is:
		// G = M1^T * w_node + M2^T * w_edge
		user->matvec_timer.start();
		ierr = MatMultTranspose(user->M1, user->g_node, user->w_temp); CHKERRQ(ierr);
		ierr = MatMultTransposeAdd(user->M2, user->g_edge, user->w_temp, G); CHKERRQ(ierr);
		user->matvec_timer.start();	

		user->objgrad_timer.stop();
		
		PetscFunctionReturn(0);
	}


	// Get the letter-wise and word-wise error of for the local sub-dataset for training
	// It includes an MAP inference by dynamic programming.
	// Input Parameters:
	// 	 fx: 				function prediction (X*w)
	// 	 labels: 		labels
	// 	 w_edgeloc: edge weights   
	// 	 user: 			user defined application context 
	// 	 seq: 			sequence information context
	// Output parameters:
	// 	 lError: 		the letter-wise error rate
	// 	 wError: 		the word-wise error rate
	PetscErrorCode get_errors(Vec &fx, Vec &labels, Vec &w_edgeloc, SeqCtx *seq, 
			AppCtx* user, PetscInt *lError, PetscInt *wError)	{
		PetscErrorCode    ierr;
		PetscReal *fx_array, *labels_array, *w_edge_array, *buf;
		PetscInt i, j, k, idx, nClass = user->nclasses;
		PetscReal **score = user->score;
		PetscReal **weight_edge = user->weight_edge;
		PetscReal **argmax = user->left, **maxVal = user->right;	// just borrow the space

		PetscFunctionBegin;
		ierr = VecGetArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecGetArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecGetArray(user->w_edgeloc, &w_edge_array); CHKERRQ(ierr);

		// Turn w_edgeloc into a 2-D array for easier indexing
		weight_edge[0] = w_edge_array;				
		for (PetscInt j=1; j < user->nclasses; j++)  
			weight_edge[j] = weight_edge[j-1] + user->nclasses;		

		ierr = PetscMalloc(user->nclasses * sizeof(PetscReal), &buf); CHKERRQ(ierr);
		*lError = *wError = 0;			

		for (PetscInt w = seq->wBegin; w < seq->wEnd; w++)		{

			PetscInt wLen = seq->wLen[w];

			// score[i][j]: score (linear discriminant value) for i-th letter and j-th class
			score[0] = fx_array;
			for (i = 1; i < wLen; i++)				
				score[i] = score[i - 1] + nClass;

			// Now use dynamic programming to find the most likely sequence output
			for (j = 0; j < nClass; j ++)
				maxVal[0][j] = score[0][j];

			for (i = 1;  i < wLen; i ++)  {
				for (j = 0; j < nClass; j++) {
					for (k = 0; k < nClass; k++)
						buf[k] = maxVal[i-1][k] + weight_edge[k][j];
					idx = std::max_element(buf, buf+nClass) - buf;
					argmax[i][j] = idx;
					maxVal[i][j] = buf[idx] + score[i][j];
				}
			}

			idx = std::max_element(maxVal[wLen-1], maxVal[wLen-1]+nClass) - maxVal[wLen-1];
			for (i = wLen-1; i >= 1; i--)  {
				maxVal[i][0] = idx;
				idx = argmax[i][idx];
			}
			maxVal[0][0] = idx;

			PetscBool wFail = PETSC_FALSE;
			for (i = 0; i < wLen; i++) {
				if ((int)(maxVal[i][0]+0.5) != (int)(*labels_array+0.5))  {
					(*lError)++;
					wFail = PETSC_TRUE;
				}
				labels_array++;
			}
			if (wFail)  (*wError)++;				
			fx_array += wLen * nClass;
		}

		fx_array -= nClass * seq->lLocalCount;
		labels_array -= seq->lLocalCount;
		ierr = VecRestoreArray(fx, &fx_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(labels, &labels_array); CHKERRQ(ierr);
		ierr = VecRestoreArray(w_edgeloc, &w_edge_array); CHKERRQ(ierr);
		ierr = PetscFree(buf); CHKERRQ(ierr);		
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
		
		// get w_node and w_edge first
		ierr = MatMult(user->M1, w, user->w_node); CHKERRQ(ierr);		
		ierr = MatMult(user->M2, w, user->w_edge); CHKERRQ(ierr);
		ierr = MatMult(user->data, user->w_node, user->fx); CHKERRQ(ierr);

		// scatter w_edge to each worker
		// ierr = VecScatterBegin(user->scatter, user->w_edge, user->w_edgeloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
		// ierr = VecScatterEnd(user->scatter, user->w_edge, user->w_edgeloc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
		ierr = Scatter(user, 0); CHKERRQ(ierr);

		// Word-wise and letter-wise error from training data
		ierr = get_errors(user->fx, user->labels, user->w_edgeloc, &user->seq, user, &lError, &wError); CHKERRQ(ierr);

		// Sum up local errors letter_wise_errors & word_wise_errors
		MPI_Allreduce(MPI_IN_PLACE, &lError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &wError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

		// If master, then print
		if (user->rank == 0)
			PetscPrintf(PETSC_COMM_SELF, "%f %f ", 
					lError*100.0/user->m, wError*100.0/user->seq.wGlobalCount);

		// We need to do the same for test set
		ierr = MatMult(user->tdata, user->w_node, user->tfx); CHKERRQ(ierr);

		// Word-wise and letter-wise error from training data
		ierr = get_errors(user->tfx, user->tlabels, user->w_edgeloc, &user->tseq, user, &lError, &wError); CHKERRQ(ierr);

		// Sum up the errors (both word wise and letter wise)
		MPI_Allreduce(MPI_IN_PLACE, &lError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &wError, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);

		// If master, then print
		if (user->rank == 0)
			PetscPrintf(PETSC_COMM_SELF, "%f %f\n", 
					lError*100.0 / user->tm, wError*100.0 / user->tseq.wGlobalCount);
		PetscFunctionReturn(0);	
	}

	// Destroy the work variables
	PetscErrorCode DestroyWorkSpace(Vec *w, AppCtx *user) {
		PetscErrorCode     ierr;

		PetscFunctionBegin;		

		ierr = MatDestroy(&user->data); CHKERRQ(ierr);
		ierr = VecDestroy(&user->labels); CHKERRQ(ierr);
		ierr = VecDestroy(&user->fx); CHKERRQ(ierr);

		ierr = MatDestroy(&user->tdata); CHKERRQ(ierr);
		ierr = VecDestroy(&user->tlabels); CHKERRQ(ierr);
		ierr = VecDestroy(&user->tfx); CHKERRQ(ierr);

		ierr = VecDestroy(w); CHKERRQ(ierr);
		ierr = VecDestroy(&user->c_node); CHKERRQ(ierr);
		ierr = VecDestroy(&user->w_node); CHKERRQ(ierr);
		ierr = VecDestroy(&user->w_edge); CHKERRQ(ierr);
		ierr = VecDestroy(&user->w_edgeloc); CHKERRQ(ierr);
		ierr = VecDestroy(&user->g_node); CHKERRQ(ierr);
		ierr = VecDestroy(&user->g_edge); CHKERRQ(ierr);
		ierr = VecDestroy(&user->g_edgeloc); CHKERRQ(ierr);
		ierr = VecDestroy(&user->w_temp); CHKERRQ(ierr);

		ierr = VecScatterDestroy(&user->scatter); CHKERRQ(ierr);

		// buffers for dynamic programming				
		ierr = PetscFree(user->marg_edge[0]); CHKERRQ(ierr);
		ierr = PetscFree(user->marg_edge); CHKERRQ(ierr);
		ierr = PetscFree(user->left[0]); CHKERRQ(ierr);
		ierr = PetscFree(user->left); CHKERRQ(ierr);
		ierr = PetscFree(user->right[0]); CHKERRQ(ierr);
		ierr = PetscFree(user->right); CHKERRQ(ierr);

		ierr = PetscFree(user->score); CHKERRQ(ierr);
		ierr = PetscFree(user->marg); CHKERRQ(ierr);
		ierr = PetscFree(user->weight_edge); CHKERRQ(ierr);
		ierr = PetscFree(user->grad_edge); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}

	// Allocate a 2-D array with 'row' number of rows, and 'col' number of columns
	PetscErrorCode make_matrix_buf(PetscInt row, PetscInt col, PetscReal ** &result) {				
		PetscReal 			*buf;	
		PetscErrorCode  ierr;

		PetscFunctionBegin;
		ierr = PetscMalloc(row * col * sizeof(PetscReal), &buf); CHKERRQ(ierr);		
		ierr = PetscMalloc(row * sizeof(PetscReal*), &result); CHKERRQ(ierr);
		result[0] = buf;		
		for (PetscInt i = 1; i < row; i ++)
			result[i] = result[i-1] + col;
		PetscFunctionReturn(0);
	}

	// Allocate the work variables for CRF loss
	PetscErrorCode AllocateWorkSpace(Vec *w, AppCtx *user) {
		PetscErrorCode     ierr;
		PetscInt  m_local, tm_local, dim_local;
		PetscInt nclasses = user->nclasses;

		PetscFunctionBegin;
		// m_local should be 26 times the number of local letters in the training data
		// dim_local should be 26 times the number of local features
		ierr = MatGetLocalSize(user->data, &m_local, &dim_local); CHKERRQ(ierr);
		ierr = MatGetLocalSize(user->tdata, &tm_local, &dim_local); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 
				(nclasses+user->dim)*nclasses, w); CHKERRQ(ierr);
		ierr = VecSetFromOptions(*w); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, m_local, PETSC_DETERMINE, &user->fx); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->fx); CHKERRQ(ierr);
		ierr = VecDuplicate(user->fx, &user->c_node); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, tm_local, PETSC_DETERMINE, &user->tfx); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->tfx); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, dim_local, PETSC_DETERMINE, &user->w_node); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->w_node); CHKERRQ(ierr);

		ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE,
				nclasses * nclasses, &user->w_edge); CHKERRQ(ierr);
		ierr = VecSetFromOptions(user->w_edge); CHKERRQ(ierr);

		ierr = VecDuplicate(user->w_node, &user->g_node); CHKERRQ(ierr);
		ierr = VecDuplicate(user->w_edge, &user->g_edge); CHKERRQ(ierr);
		ierr = VecDuplicate(*w, &user->w_temp); CHKERRQ(ierr);

		ierr = VecScatterCreateToAll(user->w_edge, &user->scatter, &user->w_edgeloc); CHKERRQ(ierr);
		ierr = VecAssemblyBegin(user->w_edgeloc); CHKERRQ(ierr);
		ierr = VecAssemblyEnd(user->w_edgeloc); CHKERRQ(ierr);

		ierr = VecScatterCreateToAll(user->g_edge, &user->scatter, &user->g_edgeloc); CHKERRQ(ierr);
		ierr = VecAssemblyBegin(user->g_edgeloc); CHKERRQ(ierr);
		ierr = VecAssemblyEnd(user->g_edgeloc); CHKERRQ(ierr);

		ierr = make_sparse_matrix(&user->M1, *w, user->w_node, 0); CHKERRQ(ierr);		
		ierr = make_sparse_matrix(&user->M2, *w, user->w_edge, user->dim * user->nclasses); CHKERRQ(ierr);

		// buffers for dynamic programming		
		SeqCtx *seq = &user->seq;		
		ierr = make_matrix_buf(nclasses, nclasses, user->marg_edge); CHKERRQ(ierr);
		ierr = make_matrix_buf(seq->maxWordLength, nclasses, user->left); CHKERRQ(ierr);
		ierr = make_matrix_buf(seq->maxWordLength, nclasses, user->right); CHKERRQ(ierr);

		ierr = PetscMalloc(seq->maxWordLength * sizeof(PetscReal*), &user->score); CHKERRQ(ierr);
		ierr = PetscMalloc(seq->maxWordLength * sizeof(PetscReal*), &user->marg); CHKERRQ(ierr);
		ierr = PetscMalloc(nclasses * sizeof(PetscReal*), &user->weight_edge); CHKERRQ(ierr);
		ierr = PetscMalloc(nclasses * sizeof(PetscReal*), &user->grad_edge); CHKERRQ(ierr);

		PetscFunctionReturn(0);
	}
}
