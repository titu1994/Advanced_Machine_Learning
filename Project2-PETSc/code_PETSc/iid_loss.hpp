#ifndef _IIDLOSS_HPP_
#define _IIDLOSS_HPP_

#include "appctx.hpp"

namespace iid_loss{
       
	PetscErrorCode LossGrad(Tao tao, Vec w, double *f, Vec G, void *ctx);

	PetscErrorCode Evaluate(Vec w, AppCtx* user);

	PetscErrorCode DestroyWorkSpace(Vec *w, AppCtx *user);

	PetscErrorCode AllocateWorkSpace(Vec *w, AppCtx *user);
}

#endif
