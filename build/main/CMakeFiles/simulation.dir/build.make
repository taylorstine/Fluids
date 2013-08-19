# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/taylor/code/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/taylor/code/build

# Include any dependencies generated for this target.
include main/CMakeFiles/simulation.dir/depend.make

# Include the progress variables for this target.
include main/CMakeFiles/simulation.dir/progress.make

# Include the compile flags for this target's objects.
include main/CMakeFiles/simulation.dir/flags.make

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o: main/CMakeFiles/simulation.dir/flags.make
main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o: /home/taylor/code/src/main/fluid_simulator.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/taylor/code/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o"
	cd /home/taylor/code/build/main && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/simulation.dir/fluid_simulator.cpp.o -c /home/taylor/code/src/main/fluid_simulator.cpp

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simulation.dir/fluid_simulator.cpp.i"
	cd /home/taylor/code/build/main && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/taylor/code/src/main/fluid_simulator.cpp > CMakeFiles/simulation.dir/fluid_simulator.cpp.i

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simulation.dir/fluid_simulator.cpp.s"
	cd /home/taylor/code/build/main && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/taylor/code/src/main/fluid_simulator.cpp -o CMakeFiles/simulation.dir/fluid_simulator.cpp.s

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.requires:
.PHONY : main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.requires

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.provides: main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.requires
	$(MAKE) -f main/CMakeFiles/simulation.dir/build.make main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.provides.build
.PHONY : main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.provides

main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.provides.build: main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o

main/CMakeFiles/simulation.dir/mesh.cpp.o: main/CMakeFiles/simulation.dir/flags.make
main/CMakeFiles/simulation.dir/mesh.cpp.o: /home/taylor/code/src/main/mesh.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/taylor/code/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object main/CMakeFiles/simulation.dir/mesh.cpp.o"
	cd /home/taylor/code/build/main && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/simulation.dir/mesh.cpp.o -c /home/taylor/code/src/main/mesh.cpp

main/CMakeFiles/simulation.dir/mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simulation.dir/mesh.cpp.i"
	cd /home/taylor/code/build/main && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/taylor/code/src/main/mesh.cpp > CMakeFiles/simulation.dir/mesh.cpp.i

main/CMakeFiles/simulation.dir/mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simulation.dir/mesh.cpp.s"
	cd /home/taylor/code/build/main && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/taylor/code/src/main/mesh.cpp -o CMakeFiles/simulation.dir/mesh.cpp.s

main/CMakeFiles/simulation.dir/mesh.cpp.o.requires:
.PHONY : main/CMakeFiles/simulation.dir/mesh.cpp.o.requires

main/CMakeFiles/simulation.dir/mesh.cpp.o.provides: main/CMakeFiles/simulation.dir/mesh.cpp.o.requires
	$(MAKE) -f main/CMakeFiles/simulation.dir/build.make main/CMakeFiles/simulation.dir/mesh.cpp.o.provides.build
.PHONY : main/CMakeFiles/simulation.dir/mesh.cpp.o.provides

main/CMakeFiles/simulation.dir/mesh.cpp.o.provides.build: main/CMakeFiles/simulation.dir/mesh.cpp.o

# Object files for target simulation
simulation_OBJECTS = \
"CMakeFiles/simulation.dir/fluid_simulator.cpp.o" \
"CMakeFiles/simulation.dir/mesh.cpp.o"

# External object files for target simulation
simulation_EXTERNAL_OBJECTS =

main/libsimulation.a: main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o
main/libsimulation.a: main/CMakeFiles/simulation.dir/mesh.cpp.o
main/libsimulation.a: main/CMakeFiles/simulation.dir/build.make
main/libsimulation.a: main/CMakeFiles/simulation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libsimulation.a"
	cd /home/taylor/code/build/main && $(CMAKE_COMMAND) -P CMakeFiles/simulation.dir/cmake_clean_target.cmake
	cd /home/taylor/code/build/main && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simulation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
main/CMakeFiles/simulation.dir/build: main/libsimulation.a
.PHONY : main/CMakeFiles/simulation.dir/build

main/CMakeFiles/simulation.dir/requires: main/CMakeFiles/simulation.dir/fluid_simulator.cpp.o.requires
main/CMakeFiles/simulation.dir/requires: main/CMakeFiles/simulation.dir/mesh.cpp.o.requires
.PHONY : main/CMakeFiles/simulation.dir/requires

main/CMakeFiles/simulation.dir/clean:
	cd /home/taylor/code/build/main && $(CMAKE_COMMAND) -P CMakeFiles/simulation.dir/cmake_clean.cmake
.PHONY : main/CMakeFiles/simulation.dir/clean

main/CMakeFiles/simulation.dir/depend:
	cd /home/taylor/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/taylor/code/src /home/taylor/code/src/main /home/taylor/code/build /home/taylor/code/build/main /home/taylor/code/build/main/CMakeFiles/simulation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : main/CMakeFiles/simulation.dir/depend

