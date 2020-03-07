cuda_add_executable(mulfft mulfft.cu)
target_include_directories(mulfft
  PUBLIC 
  ${PROJECT_SOURCE_DIR}/include 
  ${PROJECT_SOURCE_DIR}/TFHEpp/include
  )
target_compile_options(mulfft
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -arch=sm_70>
    )