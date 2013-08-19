add_executable(fluids main.cpp)
#add_library(simulation fluid_simulator.cpp mesh.cpp)

Find_Package( SDL REQUIRED )
Find_Package( OpenGL REQUIRED)

if(NOT SDL_FOUND)
  message(FATAL_ERROR "SDL was not found")
endif(NOT SDL_FOUND)

target_link_libraries(
  fluids
  math
  ${SDL_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARIES}
)

set(GLEW_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external_libs/glew-1.6.0)

install(TARGETS fluids DESTINATION ${CFD_SOURCE_DIR}/..)