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
CMAKE_SOURCE_DIR = /home/sisi/D/git/sisiwuxi/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sisi/D/git/sisiwuxi/cuda/build

# Include any dependencies generated for this target.
include 17_UVA/CMakeFiles/UVA.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 17_UVA/CMakeFiles/UVA.dir/compiler_depend.make

# Include the progress variables for this target.
include 17_UVA/CMakeFiles/UVA.dir/progress.make

# Include the compile flags for this target's objects.
include 17_UVA/CMakeFiles/UVA.dir/flags.make

17_UVA/CMakeFiles/UVA.dir/UVA.cu.o: 17_UVA/CMakeFiles/UVA.dir/flags.make
17_UVA/CMakeFiles/UVA.dir/UVA.cu.o: ../17_UVA/UVA.cu
17_UVA/CMakeFiles/UVA.dir/UVA.cu.o: 17_UVA/CMakeFiles/UVA.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/sisiwuxi/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 17_UVA/CMakeFiles/UVA.dir/UVA.cu.o"
	cd /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/sisiwuxi/cuda/17_UVA/UVA.cu -o CMakeFiles/UVA.dir/UVA.cu.o
	cd /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/sisiwuxi/cuda/17_UVA/UVA.cu -MT CMakeFiles/UVA.dir/UVA.cu.o -o CMakeFiles/UVA.dir/UVA.cu.o.d

17_UVA/CMakeFiles/UVA.dir/UVA.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/UVA.dir/UVA.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

17_UVA/CMakeFiles/UVA.dir/UVA.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/UVA.dir/UVA.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target UVA
UVA_OBJECTS = \
"CMakeFiles/UVA.dir/UVA.cu.o"

# External object files for target UVA
UVA_EXTERNAL_OBJECTS =

17_UVA/UVA: 17_UVA/CMakeFiles/UVA.dir/UVA.cu.o
17_UVA/UVA: 17_UVA/CMakeFiles/UVA.dir/build.make
17_UVA/UVA: 17_UVA/CMakeFiles/UVA.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/sisiwuxi/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable UVA"
	cd /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/UVA.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
17_UVA/CMakeFiles/UVA.dir/build: 17_UVA/UVA
.PHONY : 17_UVA/CMakeFiles/UVA.dir/build

17_UVA/CMakeFiles/UVA.dir/clean:
	cd /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA && $(CMAKE_COMMAND) -P CMakeFiles/UVA.dir/cmake_clean.cmake
.PHONY : 17_UVA/CMakeFiles/UVA.dir/clean

17_UVA/CMakeFiles/UVA.dir/depend:
	cd /home/sisi/D/git/sisiwuxi/cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/sisiwuxi/cuda /home/sisi/D/git/sisiwuxi/cuda/17_UVA /home/sisi/D/git/sisiwuxi/cuda/build /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA /home/sisi/D/git/sisiwuxi/cuda/build/17_UVA/CMakeFiles/UVA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 17_UVA/CMakeFiles/UVA.dir/depend
