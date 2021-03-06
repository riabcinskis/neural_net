cmake_minimum_required (VERSION 2.8)
set( PROJECT_NAME ann-demo )
project ( ${PROJECT_NAME} )



FIND_PACKAGE(CUDA)

IF(CUDA_FOUND)
  INCLUDE(FindCUDA)
  INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )
ENDIF(CUDA_FOUND)
 


# source files
SET ( CPP_SOURCES
	"${CMAKE_SOURCE_DIR}/src/tests.cpp"
	"${CMAKE_SOURCE_DIR}/src/ann.cpp"
	"${CMAKE_SOURCE_DIR}/src/ann-demo.cpp"


) 

SET ( CUDA_SOURCES
  "${CMAKE_SOURCE_DIR}/src/ann.cu"
)

include_directories( ${CMAKE_SOURCE_DIR}/src/extra/cuda_inc/ )


set(CMAKE_CXX_FLAGS "-O3 -fopenmp --std=c++11")

# compile java sources
#set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")

#set(CMAKE_CXX_FLAGS "-O2")
# compile cuda sources
# -Xptxas -dlcm=ca # load granularity: 128 bytes, hits L1
# -Xptxas -dlcm=cg # load granularity: 32 bytes, hits L2
IF(CUDA_FOUND)	
  LIST(APPEND CUDA_NVCC_FLAGS -lineinfo  -fmad=true -Xptxas -dlcm=ca  -gencode arch=compute_50,code=sm_50)
  CUDA_ADD_LIBRARY(cudalib ${CUDA_SOURCES} SHARED )
  TARGET_LINK_LIBRARIES(cudalib ${CUDA_LIBRARIES} )
ENDIF(CUDA_FOUND)


# compile cpp sources 
add_executable( ${PROJECT_NAME}  ${CPP_SOURCES} )

IF(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME} cudalib)
ENDIF(CUDA_FOUND)

