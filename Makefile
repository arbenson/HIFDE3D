LIBS = -llapack -lblas -lm
AR = ar
ARFLAGS = rc
CXX = g++
CXXFLAGS = -g -O3 -W -Wall -Wextra -pedantic #-std=c++0x
LDFLAGS = ${LIBS}
RANLIB = ranlib

# default rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) -Iinclude -c $< -o $@

HIF_SRC = src/global.cpp \
           src/main.cpp \
           src/ID.cpp

HMAT_SRC = src/hmat_tools/Add.cpp \
           src/hmat_tools/AdjointMultiplyMatrix.cpp \
           src/hmat_tools/AdjointMultiplyVector.cpp \
           src/hmat_tools/Compress.cpp \
           src/hmat_tools/Invert.cpp \
           src/hmat_tools/MultiplyAdjoint.cpp \
           src/hmat_tools/MultiplyMatrix.cpp \
           src/hmat_tools/MultiplyTranspose.cpp \
           src/hmat_tools/MultiplyVector.cpp \
           src/hmat_tools/TransposeMultiplyMatrix.cpp \
           src/hmat_tools/TransposeMultiplyVector.cpp \
           src/hmat_tools/Update.cpp

LIB_SRC = $(HMAT_SRC) $(HIF_SRC)


LIB_OBJ = $(LIB_SRC:.cpp=.o)

libhifde.a: ${LIB_OBJ}
	$(AR) $(ARFLAGS) $@ $(LIB_OBJ)
	$(RANLIB) $@

all: src/main.o libhifde.a
	${CXX} -o $@ $^ ${LDFLAGS}

#------------------------------------------------------
clean:
	rm -rf *~ src/*.d src/*.o *.a test

tags:
	etags include/*.hpp src/*.cpp
