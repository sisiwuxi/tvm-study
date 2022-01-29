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
include 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/compiler_depend.make

# Include the progress variables for this target.
include 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/progress.make

# Include the compile flags for this target's objects.
include 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/flags.make

4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o: 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/flags.make
4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o: 4_sum_arrays_timer/sum_arrays_timer.cu
4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o: 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o"
	cd /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer/sum_arrays_timer.cu -o CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o
	cd /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer/sum_arrays_timer.cu -MT CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o -o CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o.d

4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target sum_arrays_timer
sum_arrays_timer_OBJECTS = \
"CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o"

# External object files for target sum_arrays_timer
sum_arrays_timer_EXTERNAL_OBJECTS =

4_sum_arrays_timer/sum_arrays_timer: 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/sum_arrays_timer.cu.o
4_sum_arrays_timer/sum_arrays_timer: 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/build.make
4_sum_arrays_timer/sum_arrays_timer: 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable sum_arrays_timer"
	cd /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sum_arrays_timer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/build: 4_sum_arrays_timer/sum_arrays_timer
.PHONY : 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/build

4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/clean:
	cd /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer && $(CMAKE_COMMAND) -P CMakeFiles/sum_arrays_timer.dir/cmake_clean.cmake
.PHONY : 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/clean

4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/depend:
	cd /home/sisi/D/git/CUDA_Freshman && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer /home/sisi/D/git/CUDA_Freshman/4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 4_sum_arrays_timer/CMakeFiles/sum_arrays_timer.dir/depend

