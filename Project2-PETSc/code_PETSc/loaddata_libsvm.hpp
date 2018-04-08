#ifndef _LOADDATA_LIBSVM_HPP_
#define _LOADDATA_LIBSVM_HPP_

#include "appctx.hpp"

PetscErrorCode parse_file(FILE* input, AppCtx_file* user);

PetscErrorCode parse_line(char* line, PetscInt* idxs, PetscReal* vals, PetscReal* label, int m, AppCtx_file* user);

PetscErrorCode assemble_matrix(FILE* input, PetscInt begin, PetscInt end, AppCtx_file* user);

PetscErrorCode fill_arrays_uni(FILE* input, AppCtx_file* user);

PetscErrorCode reparse_file(FILE* input, PetscInt* diag_nnz, PetscInt* offdiag_nnz, AppCtx_file* user);

PetscErrorCode fill_arrays_parallel(FILE* input, AppCtx_file* user);

PetscErrorCode LoadDataLibsvm(AppCtx *user_main);

#endif
