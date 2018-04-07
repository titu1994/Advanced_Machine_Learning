#include "loaddata_libsvm.hpp"

// Trucate the memory space by retaining len
PetscErrorCode truncate_memory(PetscInt **buf, size_t len)  {
	PetscFunctionBegin;

	PetscErrorCode ierr;
	void *result = PETSC_NULL;
	ierr = PetscMalloc(len, &result); CHKERRQ(ierr);
	ierr = PetscMemcpy(result, *buf, len); CHKERRQ(ierr);
	ierr = PetscFree(*buf); CHKERRQ(ierr);
	*buf = (PetscInt *)result;
	PetscFunctionReturn(0);
}

// Englarge the memory space from len_old to len_new
PetscErrorCode expand_memory(PetscInt **buf, size_t len_old, size_t len_new)  {
	PetscFunctionBegin;
	PetscErrorCode ierr;
	void *result = PETSC_NULL;
	ierr = PetscMalloc(len_new, &result); CHKERRQ(ierr);
	ierr = PetscMemcpy(result, *buf, len_old); CHKERRQ(ierr);
	ierr = PetscFree(*buf); CHKERRQ(ierr);
	*buf = (PetscInt *)result;
	PetscFunctionReturn(0);
}


// The job of this function, run only by the master node, is to calculate:
// 1. number of lines/letters (fctx->m)
// 2. number of nnz for all lines (fctx->nnz_array)
// 3. max number of nonzero features among all letters (fctx->maxnnz)
// 4. check if minimum index is 0 or 1
// 5. Max #char of each line in the input file (fctx->maxlen)
// 6. number of letters for each word
PetscErrorCode parse_file(FILE* input, AppCtx_file* fctx) {
  PetscErrorCode ierr;
  size_t        len = 0;				// buffer length
	PetscInt			llen = 0;				// number of characters in a line
  char          *line = 0;
	char					*pidx;
	PetscInt      m_guess = 128; // guess of #letters/#lines
	PetscInt      w_guess = 128;	// guess of #words
	PetscInt			wID, pre_wID;		// ID of the current word and previous word
	SeqCtx				*seq = fctx->seq;
  PetscFunctionBegin;
   
	fctx->nnz_array = seq->wLen = PETSC_NULL;
	fctx->maxlen = fctx->maxnnz = fctx->m = seq->wGlobalCount = 0;
	fctx->dim = -1;
	pre_wID = -1;		// ID (qid) of the last word. No word can have qid = -1.
	seq->maxWordLength = -1;
	
	// Guess the number of training letters in the whole dataset
	//  (note, this function is only run by the master node)
  ierr = PetscMalloc(m_guess * sizeof(PetscInt), &fctx->nnz_array); CHKERRQ(ierr);
	ierr = PetscMalloc(w_guess * sizeof(PetscInt), &seq->wLen); CHKERRQ(ierr);
	
  // parse input file line by line
  while ((llen = getline(&line, &len, input)) !=  -1) {

		fctx->maxlen = std::max(fctx->maxlen, llen);	// max #char in each input line

    // Skip label
		strtok(line, " \t");
		strtok(NULL, ":");		// Skip the "qid"		
		pidx = strtok(NULL, " \t");		// Read and parse the qid number
		wID = (PetscInt)strtol(pidx, NULL, 10);
		if (wID != pre_wID) 		{						// a new word starts
			if (seq->wGlobalCount >= w_guess) {		// enlarge the array of #letter (per word)
				expand_memory(&seq->wLen, w_guess*sizeof(PetscInt), w_guess*2*sizeof(PetscInt));
				w_guess *= 2;
			}
			
			seq->wLen[seq->wGlobalCount++] = 0;
			pre_wID = wID;
			if (seq->wGlobalCount > 1)
				seq->maxWordLength = std::max(seq->maxWordLength, seq->wLen[seq->wGlobalCount-2]);
		}
		seq->wLen[seq->wGlobalCount-1]++;
		
		PetscInt nnz = 1; 	// bias   
    while (1) {  // Now process the rest of the line (features)
      // parse id:val pairs of feature
      char* pidx = strtok(NULL, ":");
      char* pval = strtok(NULL, " \t");

      if (pval == NULL)
        break;
      PetscInt idx = (PetscInt) strtol(pidx, NULL, 10);
			fctx->dim = std::max(fctx->dim, idx);
      nnz++;
    }

    if (fctx->m >= m_guess) {				// enlarge the array of #nnz
			expand_memory(&fctx->nnz_array, m_guess*sizeof(PetscInt), m_guess*2*sizeof(PetscInt));
			m_guess *= 2;
    }
		
		// Remember the maximum number of nonzeros in any row 
		fctx->maxnnz = std::max(fctx->maxnnz, nnz);
    fctx->nnz_array[fctx->m++] = nnz;
  }

  if (line)    free(line);
	seq->maxWordLength = std::max(seq->maxWordLength, seq->wLen[seq->wGlobalCount-1]);
	fctx->dim++;	// dimensions = max idx + 1 because feature index starts from 1 in files
								// and we always add a bias feature (1)

  // Adjust to finalize the array of size
	ierr = truncate_memory(&fctx->nnz_array, fctx->m*sizeof(PetscInt)); CHKERRQ(ierr);
	ierr = truncate_memory(&seq->wLen, seq->wGlobalCount*sizeof(PetscInt)); CHKERRQ(ierr);
	PetscFunctionReturn(0);
}



// Given a line, retrieve the label, and the idx:val pair of features
PetscErrorCode parse_line(char* line,
                PetscInt* idxs, 
                PetscScalar* vals, 
                PetscScalar* label, 
								PetscInt m,
                AppCtx_file* fctx) {
	PetscFunctionBegin;

  // Read the label
  char *plabel = strtok(line, " \t");
  *label = strtol(plabel, NULL, 10);
	strtok(NULL, ":");			// skip "qid"
	strtok(NULL, " \t");		// skip word id

  // process the line
  PetscInt nnz = 0;
  while (1) {
    char* pidx = strtok(NULL, ":");
    char* pval = strtok(NULL, " \t");
    if (pval == NULL)
      break;
		idxs[nnz] = (PetscInt)strtol(pidx, NULL, 10)-1;
    vals[nnz] = strtod(pval, NULL);
    nnz++;
  }  
  vals[nnz]=1.0;
  idxs[nnz]=fctx->dim-1;
  nnz++;
	PetscFunctionReturn(0);
}


// For uniprocessor, fill in the values of the data matrix and label vector
// only take lines from begin to end-1
PetscErrorCode assemble_matrix(FILE* input, 
                               PetscInt begin, 
                               PetscInt end, 
                               AppCtx_file* fctx) {  
  PetscErrorCode ierr;
  size_t         len = fctx->maxlen;
  char*          line = (char *) malloc(len*sizeof(char));
  PetscInt       m = 0;
  PetscInt       ii = 0;
  PetscScalar    label = 0;
  PetscScalar*   vals = 0;
  PetscInt*      idxs = 0;

  PetscFunctionBegin;

  ierr = PetscMalloc(fctx->maxnnz*sizeof(PetscScalar), &vals);	CHKERRQ(ierr);
  ierr = PetscMalloc(fctx->maxnnz*sizeof(PetscInt), &idxs);	CHKERRQ(ierr);

  while (getline(&line, &len, input) !=  -1) {
    //skip the lines for which this processor is not responsible
    if (m < begin) {
      m++;  continue;
    } else if (m >= end) 
      break;
    parse_line(line, idxs, vals, &label, m, fctx);
    ierr = VecSetValues(*fctx->labels, 1, &m, &label, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(*fctx->data, 1, &m, fctx->nnz_array[m], idxs, vals, INSERT_VALUES);CHKERRQ(ierr);
    m++;    ii++;
  }

  free(line);
  ierr = PetscFree(vals); CHKERRQ(ierr);
  ierr = PetscFree(idxs); CHKERRQ(ierr);

  ierr = VecAssemblyBegin(*fctx->labels); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(*fctx->labels); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*fctx->data, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*fctx->data, MAT_FINAL_ASSEMBLY);	CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// For uniprocessor, create and fill in the data matrix
PetscErrorCode fill_arrays_uni(FILE* input, AppCtx_file* fctx) {
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  // Allocate space for the labels
  ierr = VecCreateSeq(PETSC_COMM_WORLD, fctx->m, fctx->labels);CHKERRQ(ierr);  
  ierr = VecSetFromOptions(*(fctx->labels));CHKERRQ(ierr);

  // Allocate space for the data matrix
  ierr = MatCreate(PETSC_COMM_WORLD, fctx->data);CHKERRQ(ierr);
  ierr = MatSetSizes(*(fctx->data), PETSC_DECIDE, PETSC_DECIDE, fctx->m, fctx->dim);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*(fctx->data));CHKERRQ(ierr);

  ierr = MatSetType(*(fctx->data), MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*(fctx->data), 0, fctx->nnz_array); CHKERRQ(ierr);

	// Fill in the values of the nonzero entries of the data matrix
	assemble_matrix(input, fctx->seq->lBegin, fctx->seq->lEnd, fctx);
  PetscFunctionReturn(0);
}


// For multi-processor, collect statistics (nnz) related to diagonal and off-diagonal
PetscErrorCode reparse_file(FILE* input, 
                            PetscInt* diag_nnz, 
                            PetscInt* offdiag_nnz, 
                            AppCtx_file* fctx) {
  PetscFunctionBegin;
  size_t len = fctx->maxlen;
  char* line = (char *) malloc(len*sizeof(char));

	PetscInt begin, end, localsize = fctx->seq->lLocalCount;
	MPI_Scan(&localsize, &end, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  begin = end - localsize;

	// Compute the range of columns for my process
  PetscInt cbegin, cend, clocalsize = PETSC_DECIDE;
  PetscSplitOwnership(PETSC_COMM_WORLD, &clocalsize, &fctx->dim);
  MPI_Scan(&clocalsize, &cend, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  cbegin = cend - clocalsize;
	
	PetscInt m = 0;
  PetscInt ii = 0;

  for (PetscInt i = 0; i<localsize; i++)
    diag_nnz[i] = 0;

  while (getline(&line, &len, input) !=  -1) {
    //skip the lines for which this processor is not responsible
    if (m < begin) {
      m++;  
      continue;
    } else if (m >= end)
      break;
        
    strtok(line, " \t");		// skip label
		strtok(NULL, ":");			// skip "qid"
		strtok(NULL, " \t");		// skip word id

    // Now process the rest of the line
    while (1) {
      char* pidx = strtok(NULL, ":");
      char* pval = strtok(NULL, " \t");

      if (pval == NULL)
        break;
      PetscInt idx = (PetscInt) strtol(pidx, NULL, 10) - 1;
      if (idx >= cbegin && idx < cend)
        diag_nnz[ii]++;
    }
    
		if (fctx->dim-1 >= cbegin && fctx->dim-1 < cend)
        diag_nnz[ii]++;
    offdiag_nnz[ii] = fctx->nnz_array[m] - diag_nnz[ii];
    m++;    ii++;
  }
  if (line)    
    free(line);
  PetscFunctionReturn(0);
}


PetscErrorCode fill_arrays_parallel(FILE* input, 
                                    AppCtx_file* fctx) {
	PetscFunctionBegin;
  PetscErrorCode ierr;
	SeqCtx *seq = fctx->seq;

  // Create labels vector. Local #elements = local #letters
  ierr = VecCreateMPI(PETSC_COMM_WORLD, seq->lLocalCount, PETSC_DETERMINE, fctx->labels);CHKERRQ(ierr);
  ierr = VecSetFromOptions(*(fctx->labels));CHKERRQ(ierr);

	// Create data matrix. Local #rows = local #letters.  Leave local #column to PETSc to decide.
	ierr = MatCreate(PETSC_COMM_WORLD, fctx->data);CHKERRQ(ierr);
	ierr = MatSetSizes(*(fctx->data), seq->lLocalCount, PETSC_DECIDE, fctx->m, fctx->dim); CHKERRQ(ierr);
  ierr = MatSetFromOptions(*(fctx->data));CHKERRQ(ierr);
  ierr = MatSetType(*(fctx->data), MATMPIAIJ); CHKERRQ(ierr);    

	// Allocate space for the data.
	// diag_nnz/offdiag_nnz compute the #elements in the diagonal/offdiagonal columns for each row
  PetscInt *diag_nnz, *offdiag_nnz;
	ierr = PetscMalloc(seq->lLocalCount*sizeof(PetscInt), &diag_nnz); CHKERRQ(ierr);
	ierr = PetscMalloc(seq->lLocalCount*sizeof(PetscInt), &offdiag_nnz); CHKERRQ(ierr);

	// Compute diag_nnz and offdiag_nnz
	reparse_file(input, diag_nnz, offdiag_nnz, fctx);
  rewind(input);

	// Really allocate the memory for the distributed sparse matrix using diag_nnz and offdiag_nnz
  ierr = MatMPIAIJSetPreallocation(*(fctx->data), 0, diag_nnz, 0, offdiag_nnz); CHKERRQ(ierr);
	ierr = PetscFree(diag_nnz); CHKERRQ(ierr);
  ierr = PetscFree(offdiag_nnz); CHKERRQ(ierr);

	// Fill in the values of the nonzero entries of the data matrix
	assemble_matrix(input, seq->lBegin, seq->lEnd, fctx);
	PetscFunctionReturn(0);
}

PetscErrorCode ParseStats(AppCtx_file *fctx, AppCtx *user, char const *prefix) {

	PetscFunctionBegin;
	PetscErrorCode ierr;
	SeqCtx *seq = fctx->seq;

	PetscPrintf(PETSC_COMM_WORLD, "Parsing %s data: %s\n", prefix, fctx->data_path);
	
	ierr = PetscFOpen(PETSC_COMM_SELF, fctx->data_path, "r", &fctx->data_file); CHKERRQ(ierr);	
	if (user->rank == 0) {
		parse_file(fctx->data_file, fctx);
		rewind(fctx->data_file);   // Set file pointer to beginning of file		
	}

	MPI_Bcast(&fctx->dim, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&fctx->m, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&fctx->maxnnz, 1, MPIU_INT, 0, PETSC_COMM_WORLD);    // slave nodes need it to load data
	MPI_Bcast(&fctx->maxlen, 1, MPIU_INT, 0, PETSC_COMM_WORLD);    // slave nodes need it to load data
	MPI_Bcast(&seq->wGlobalCount, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&seq->maxWordLength, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
	if (user->rank != 0)	{
		ierr = PetscMalloc(fctx->m * sizeof(PetscInt), &fctx->nnz_array); CHKERRQ(ierr);
		ierr = PetscMalloc(seq->wGlobalCount*sizeof(PetscInt), &seq->wLen); CHKERRQ(ierr);
	}
	MPI_Bcast(fctx->nnz_array, fctx->m, MPIU_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(seq->wLen, seq->wGlobalCount, MPIU_INT, 0, PETSC_COMM_WORLD);

	// Now each node uses the wLen and wGlobalCount information to determine the local word/letter range
	PetscInt i, chunkSize = seq->wGlobalCount / user->size;	// chunk = (#words divided by #processes)
	seq->wBegin = user->rank * chunkSize;	// each process takes one chunk
	seq->wEnd = chunkSize + seq->wBegin;
	if(user->rank == user->size - 1)
		seq->wEnd = seq->wGlobalCount;  // last process always takes the remainder words
	// Now figure out the range of letters at each process
	seq->lBegin = 0;
	// First sum up the length of all words belonging to proceding processes
	for (i = 0; i < seq->wBegin; i++)	
		seq->lBegin += seq->wLen[i];
	seq->lEnd = seq->lBegin;
	while (i < seq->wEnd)	// now sum up the length of all local words
		seq->lEnd += seq->wLen[i++];
	seq->lLocalCount = seq->lEnd - seq->lBegin; // total #letters of local words	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LoadDataLibsvm"
// Load from data files in LIBSVM format
// Always add bias
PetscErrorCode LoadDataLibsvm(AppCtx *user) {
  PetscErrorCode    ierr;
	AppCtx_file       fctx, tfctx;	// file context for training and testing
  PetscInt					cmin, cmax;  

  PetscFunctionBegin;

  // First parse the training and test data file to collect statistics
	fctx.data = &fctx.data_parts;  
	fctx.labels = &user->labels;	
	fctx.data_path = user->data_path;
	fctx.seq = &user->seq;
  tfctx.data = &tfctx.data_parts;  
	tfctx.labels = &user->tlabels;
	tfctx.data_path = user->tdata_path;
	tfctx.seq = &user->tseq;

  // All nodes parse the training data file to collect statistics
  // Master node broadcasts the statistics of the training and test set
	ierr = ParseStats(&fctx, user, "training"); CHKERRQ(ierr);
	
	// All nodes parse the test data file to collect statistics
	ierr = ParseStats(&tfctx, user, "test"); CHKERRQ(ierr);
	
	tfctx.dim = fctx.dim = std::max(fctx.dim, tfctx.dim);
	
  // Really start to load the training data
  // PetscPrintf(PETSC_COMM_WORLD, "Start loading the training data\n");
	PetscErrorCode (*fill_arrays)(FILE*, AppCtx_file*);
	fill_arrays = user->size == 1 ? fill_arrays_uni : fill_arrays_parallel;

	fill_arrays(fctx.data_file, &fctx);		// training data
	ierr = PetscFree(fctx.nnz_array); CHKERRQ(ierr);
	PetscFClose(PETSC_COMM_SELF, fctx.data_file);

	fill_arrays(tfctx.data_file, &tfctx);	// test data
	ierr = PetscFree(tfctx.nnz_array); CHKERRQ(ierr);
	PetscFClose(PETSC_COMM_SELF, tfctx.data_file);

	PetscScalar tmp;
	ierr = VecMin(user->labels, PETSC_NULL, &tmp); CHKERRQ(ierr);
	cmin = (int)(tmp+0.5);
	ierr = VecMax(user->labels, PETSC_NULL, &tmp); CHKERRQ(ierr);
	cmax = (int)(tmp+0.5);
  user->nclasses = cmax - cmin + 1;
  
  if (cmin != 0) {
    ierr = VecShift(user->labels, -cmin);CHKERRQ(ierr);
    ierr = VecShift(user->tlabels, -cmin);CHKERRQ(ierr);
  }

  ierr = MatGetSize(fctx.data_parts, &user->m, &user->dim); CHKERRQ(ierr);
	// Replicate the training data for #class times (logically)
  ierr = MatCreateMAIJ(fctx.data_parts, user->nclasses, &user->data);CHKERRQ(ierr);
	ierr = MatDestroy(&fctx.data_parts); CHKERRQ(ierr);

  ierr = MatGetSize(tfctx.data_parts, &user->tm, &user->tdim); CHKERRQ(ierr);
	// Replicate the test data for #class times (logically)
  ierr = MatCreateMAIJ(tfctx.data_parts, user->nclasses, &user->tdata);CHKERRQ(ierr);
  ierr = MatDestroy(&tfctx.data_parts); CHKERRQ(ierr);
	PetscPrintf(PETSC_COMM_WORLD, "Data loading completed\n");
	
//	PetscPrintf(PETSC_COMM_SELF, "rank=%D, wBegin=%D, wEnd=%D, twBegin=%D, twEnd=%D\n",
//								user->rank, user->seq.wBegin, user->seq.wEnd, 
//								user->tseq.wBegin, user->tseq.wEnd);
	PetscFunctionReturn(0);
}
