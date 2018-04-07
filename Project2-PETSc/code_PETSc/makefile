all: seq_train

CFLAGS		 = 
FFLAGS		 = 
CPPFLAGS   = 
FPPFLAGS   =
LOCDIR		 =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
	

ALL_OBJECTS = iid_loss.o crf_loss.o appctx.o loaddata_libsvm.o timer.o 

seq_train: seq_train.o ${ALL_OBJECTS}  chkopts
	-${CLINKER} -o seq_train seq_train.o ${ALL_OBJECTS} ${PETSC_TAO_LIB}

demo: demo_PETSc.o  chkopts
	-${CLINKER} -o demo_PETSc demo_PETSc.o ${PETSC_TAO_LIB}

clean::
		${RM} seq_train seq_train.o
