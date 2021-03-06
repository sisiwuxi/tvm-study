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
include 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/compiler_depend.make

# Include the progress variables for this target.
include 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/progress.make

# Include the compile flags for this target's objects.
include 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/flags.make

21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o: 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/flags.make
21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o: ../21_sum_array_offset_unrolling/sum_array_offset_unrolling.cu
21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o: 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/sisiwuxi/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o"
	cd /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/sisiwuxi/cuda/21_sum_array_offset_unrolling/sum_array_offset_unrolling.cu -o CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o
	cd /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/sisiwuxi/cuda/21_sum_array_offset_unrolling/sum_array_offset_unrolling.cu -MT CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o -o CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o.d

21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sum_array_offset_unrolling
sum_array_offset_unrolling_OBJECTS = \
"CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o"

# External object files for target sum_array_offset_unrolling
sum_array_offset_unrolling_EXTERNAL_OBJECTS =

21_sum_array_offset_unrolling/sum_array_offset_unrolling: 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/sum_array_offset_unrolling.cu.o
21_sum_array_offset_unrolling/sum_array_offset_unrolling: 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/build.make
21_sum_array_offset_unrolling/sum_array_offset_unrolling: 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/sisiwuxi/cuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable sum_array_offset_unrolling"
	cd /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sum_array_offset_unrolling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/build: 21_sum_array_offset_unrolling/sum_array_offset_unrolling
.PHONY : 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/build

21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/clean:
	cd /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling && $(CMAKE_COMMAND) -P CMakeFiles/sum_array_offset_unrolling.dir/cmake_clean.cmake
.PHONY : 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/clean

21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/depend:
	cd /home/sisi/D/git/sisiwuxi/cuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/sisiwuxi/cuda /home/sisi/D/git/sisiwuxi/cuda/21_sum_array_offset_unrolling /home/sisi/D/git/sisiwuxi/cuda/build /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling /home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 21_sum_array_offset_unrolling/CMakeFiles/sum_array_offset_unrolling.dir/depend

