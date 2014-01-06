LIBS = -llapack -lblas -lm
AR = ar
ARFLAGS = rc
CXX = g++
CXXFLAGS = -g -O3 -Wall -Wextra -pedantic
LDFLAGS = ${LIBS}
RANLIB = ranlib
DEFINES = -DRESTRICT=__restrict__
INCLUDES = include

# default rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) -I$(INCLUDES) -c $< -o $@

HIF_SRC = src/global.cpp \
          src/InterpDecomp.cpp \
          src/Factor.cpp \
          src/Apply.cpp \
          src/main.cpp

HMAT_SRC = src/hmat_tools/Add.cpp \
           src/hmat_tools/Compress.cpp \
           src/hmat_tools/Update.cpp \
           src/hmat_tools/MultiplyVector.cpp \
           src/hmat_tools/AdjointMultiplyVector.cpp \
           src/hmat_tools/MultiplyTranspose.cpp \
           src/hmat_tools/MultiplyAdjoint.cpp \
           src/hmat_tools/MultiplyMatrix.cpp \
           src/hmat_tools/TransposeMultiplyVector.cpp \
           src/hmat_tools/TransposeMultiplyMatrix.cpp \
           src/hmat_tools/AdjointMultiplyMatrix.cpp \
           src/hmat_tools/Invert.cpp

LIB_SRC = $(HMAT_SRC) $(HIF_SRC)
LIB_OBJ = $(LIB_SRC:.cpp=.o)

libhifde.a: ${LIB_OBJ}
	$(AR) $(ARFLAGS) $@ $(LIB_OBJ)
	$(RANLIB) $@

all: src/main.o libhifde.a
	${CXX} -o hifde3d $^ ${LDFLAGS}

#------------------------------------------------------
clean:
	rm -rf *~ src/*.d src/*.o src/hmat_tools/*.o *.a hifde3d

tags:
	etags include/*.hpp src/*.cpp
