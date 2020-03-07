file(GLOB FFHEE_sources "*.c*")


add_library(FFHEE
  ${FFHEE_sources}
  )
target_include_directories(FFHEE
  PUBLIC ${PROJECT_SOURCE_DIR}/include 
  )