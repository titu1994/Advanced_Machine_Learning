#ifndef _CRFLOSS_HPP_
#define _CRFLOSS_HPP_

#include "appctx.hpp"

namespace crf_loss{
  
	PetscErrorCode LossGrad(Tao tao, Vec w, double *f, Vec G, void *ctx);

	PetscErrorCode Evaluate(Vec w, AppCtx* user);

	PetscErrorCode DestroyWorkSpace(Vec *w, AppCtx *user);

	PetscErrorCode AllocateWorkSpace(Vec *w, AppCtx *user);

	PetscErrorCode Scatter(AppCtx *user, int op);
}

#endif
