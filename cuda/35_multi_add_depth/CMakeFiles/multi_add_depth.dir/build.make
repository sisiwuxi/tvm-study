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
include 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/compiler_depend.make

# Include the progress variables for this target.
include 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/progress.make

# Include the compile flags for this target's objects.
include 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/flags.make

35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o: 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/flags.make
35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o: 35_multi_add_depth/multi_add_depth.cu
35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o: 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o"
	cd /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth/multi_add_depth.cu -o CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o
	cd /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth && /usr/local/cuda-9.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth/multi_add_depth.cu -MT CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o -o CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o.d

35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target multi_add_depth
multi_add_depth_OBJECTS = \
"CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o"

# External object files for target multi_add_depth
multi_add_depth_EXTERNAL_OBJECTS =

35_multi_add_depth/multi_add_depth: 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/multi_add_depth.cu.o
35_multi_add_depth/multi_add_depth: 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/build.make
35_multi_add_depth/multi_add_depth: 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sisi/D/git/CUDA_Freshman/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable multi_add_depth"
	cd /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/multi_add_depth.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
35_multi_add_depth/CMakeFiles/multi_add_depth.dir/build: 35_multi_add_depth/multi_add_depth
.PHONY : 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/build

35_multi_add_depth/CMakeFiles/multi_add_depth.dir/clean:
	cd /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth && $(CMAKE_COMMAND) -P CMakeFiles/multi_add_depth.dir/cmake_clean.cmake
.PHONY : 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/clean

35_multi_add_depth/CMakeFiles/multi_add_depth.dir/depend:
	cd /home/sisi/D/git/CUDA_Freshman && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth /home/sisi/D/git/CUDA_Freshman /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth /home/sisi/D/git/CUDA_Freshman/35_multi_add_depth/CMakeFiles/multi_add_depth.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : 35_multi_add_depth/CMakeFiles/multi_add_depth.dir/depend
