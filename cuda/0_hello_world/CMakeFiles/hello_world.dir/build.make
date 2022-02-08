# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.21.0/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.21.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sisi/D/git/CUDA_Freshman

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sisi/D/git/CUDA_Freshman

# Include any dependencies generated for this target.
include 0_hello_world/CMakeFiles/hello_world.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 0_hello_world/CMakeFiles/hello_world.dir/compiler_depend.make

# Include the progress variables for this target.
include 0_hello_world/CMakeFiles/hello_world.dir/progress.make

# Include the compile flags for this target's objects.
include 0_hello_world/CMakeFiles/hello_world.dir/flags.make

0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o: 0_hello_world/CMakeFiles/hello_world.dir/flags.make
0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o: 0_hello_world/hello_world.cu
0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o: 0_hello_world/CMakeFiles/hello_world.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o"
	cd /home/sisi/D/git/CUDA_Freshman/0_hello_world && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/CUDA_Freshman/0_hello_world/hello_world.cu -o CMakeFiles/hello_world.dir/hello_world.cu.o
	cd /home/sisi/D/git/CUDA_Freshman/0_hello_world && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/CUDA_Freshman/0_hello_world/hello_world.cu -MT CMakeFiles/hello_world.dir/hello_world.cu.o -o CMakeFiles/hello_world.dir/hello_world.cu.o.d

0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hello_world.dir/hello_world.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hello_world.dir/hello_world.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target hello_world
hello_world_OBJECTS = \
"CMakeFiles/hello_world.dir/hello_world.cu.o"

# External object files for target hello_world
hello_world_EXTERNAL_OBJECTS =

0_hello_world/hello_world: 0_hello_world/CMakeFiles/hello_world.dir/hello_world.cu.o
0_hello_world/hello_world: 0_hello_world/CMakeFiles/hello_world.dir/build.make
0_hello_world/hello_world: 0_hello_world/CMakeFiles/hello_world.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable hello_world"
	cd /home/sisi/D/git/CUDA_Freshman/0_hello_world && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_world.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
0_hello_world/CMakeFiles/hello_world.dir/build: 0_hello_world/hello_world
.PHONY : 0_hello_world/CMakeFiles/hello_world.dir/build

0_hello_world/CMakeFiles/hello_world.dir/clean:
	cd /home/sisi/D/git/CUDA_Freshman/0_hello_world && $(CMAKE_COMMAND) -P CMakeFiles/hello_world.dir/cmake_clean.cmake
.PHONY : 0_hello_world/CMakeFiles/hello_world.dir/clean

0_hello_world/CMakeFiles/hello_world.dir/depend:
	cd /home/sisi/D/git/CUDA_Freshman && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/0_hello_world /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/0_hello_world /home/sisi/D/git/CUDA_Freshman/0_hello_world/CMakeFiles/hello_world.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 0_hello_world/CMakeFiles/hello_world.dir/depend
