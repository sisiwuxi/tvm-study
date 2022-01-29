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
include 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/compiler_depend.make

# Include the progress variables for this target.
include 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/progress.make

# Include the compile flags for this target's objects.
include 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/flags.make

23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o: 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/flags.make
23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o: 23_sum_array_uniform_memory/sum_arrays_uniform_memory.cu
23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o: 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o"
	cd /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory/sum_arrays_uniform_memory.cu -o CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o
	cd /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory/sum_arrays_uniform_memory.cu -MT CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o -o CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o.d

23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sum_arrays_uniform_memory
sum_arrays_uniform_memory_OBJECTS = \
"CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o"

# External object files for target sum_arrays_uniform_memory
sum_arrays_uniform_memory_EXTERNAL_OBJECTS =

23_sum_array_uniform_memory/sum_arrays_uniform_memory: 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/sum_arrays_uniform_memory.cu.o
23_sum_array_uniform_memory/sum_arrays_uniform_memory: 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/build.make
23_sum_array_uniform_memory/sum_arrays_uniform_memory: 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable sum_arrays_uniform_memory"
	cd /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sum_arrays_uniform_memory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/build: 23_sum_array_uniform_memory/sum_arrays_uniform_memory
.PHONY : 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/build

23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/clean:
	cd /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory && $(CMAKE_COMMAND) -P CMakeFiles/sum_arrays_uniform_memory.dir/cmake_clean.cmake
.PHONY : 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/clean

23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/depend:
	cd /home/sisi/D/git/CUDA_Freshman && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory /home/sisi/D/git/CUDA_Freshman/23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 23_sum_array_uniform_memory/CMakeFiles/sum_arrays_uniform_memory.dir/depend

