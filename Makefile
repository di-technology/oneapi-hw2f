CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

USM_EXE_NAME = hw2f-usm
USM_SOURCES = src/hw2f-usm.cpp

all: build_usm

build_usm:
	$(CXX) $(CXXFLAGS) -o $(USM_EXE_NAME) $(USM_SOURCES)  -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl -ltbb

run_usm: 
	./$(USM_EXE_NAME)

clean: 
	rm -f $(USM_EXE_NAME)
