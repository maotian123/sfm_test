# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /home/maotian/.local/lib/python3.6/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/maotian/.local/lib/python3.6/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/maotian/Sometest/sfm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/maotian/Sometest/sfm/build

# Include any dependencies generated for this target.
include CMakeFiles/sfm_node.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sfm_node.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sfm_node.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sfm_node.dir/flags.make

CMakeFiles/sfm_node.dir/src/sfm_node.o: CMakeFiles/sfm_node.dir/flags.make
CMakeFiles/sfm_node.dir/src/sfm_node.o: ../src/sfm_node.cpp
CMakeFiles/sfm_node.dir/src/sfm_node.o: CMakeFiles/sfm_node.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/maotian/Sometest/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sfm_node.dir/src/sfm_node.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sfm_node.dir/src/sfm_node.o -MF CMakeFiles/sfm_node.dir/src/sfm_node.o.d -o CMakeFiles/sfm_node.dir/src/sfm_node.o -c /home/maotian/Sometest/sfm/src/sfm_node.cpp

CMakeFiles/sfm_node.dir/src/sfm_node.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sfm_node.dir/src/sfm_node.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/maotian/Sometest/sfm/src/sfm_node.cpp > CMakeFiles/sfm_node.dir/src/sfm_node.i

CMakeFiles/sfm_node.dir/src/sfm_node.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sfm_node.dir/src/sfm_node.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/maotian/Sometest/sfm/src/sfm_node.cpp -o CMakeFiles/sfm_node.dir/src/sfm_node.s

# Object files for target sfm_node
sfm_node_OBJECTS = \
"CMakeFiles/sfm_node.dir/src/sfm_node.o"

# External object files for target sfm_node
sfm_node_EXTERNAL_OBJECTS =

sfm_node: CMakeFiles/sfm_node.dir/src/sfm_node.o
sfm_node: CMakeFiles/sfm_node.dir/build.make
sfm_node: /usr/local/lib/libopencv_dnn.so.3.4.5
sfm_node: /usr/local/lib/libopencv_ml.so.3.4.5
sfm_node: /usr/local/lib/libopencv_objdetect.so.3.4.5
sfm_node: /usr/local/lib/libopencv_shape.so.3.4.5
sfm_node: /usr/local/lib/libopencv_stitching.so.3.4.5
sfm_node: /usr/local/lib/libopencv_superres.so.3.4.5
sfm_node: /usr/local/lib/libopencv_videostab.so.3.4.5
sfm_node: /usr/local/lib/libopencv_viz.so.3.4.5
sfm_node: /usr/local/lib/libopencv_calib3d.so.3.4.5
sfm_node: /usr/local/lib/libopencv_features2d.so.3.4.5
sfm_node: /usr/local/lib/libopencv_flann.so.3.4.5
sfm_node: /usr/local/lib/libopencv_highgui.so.3.4.5
sfm_node: /usr/local/lib/libopencv_photo.so.3.4.5
sfm_node: /usr/local/lib/libopencv_video.so.3.4.5
sfm_node: /usr/local/lib/libopencv_videoio.so.3.4.5
sfm_node: /usr/local/lib/libopencv_imgcodecs.so.3.4.5
sfm_node: /usr/local/lib/libopencv_imgproc.so.3.4.5
sfm_node: /usr/local/lib/libopencv_core.so.3.4.5
sfm_node: CMakeFiles/sfm_node.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/maotian/Sometest/sfm/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sfm_node"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sfm_node.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sfm_node.dir/build: sfm_node
.PHONY : CMakeFiles/sfm_node.dir/build

CMakeFiles/sfm_node.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sfm_node.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sfm_node.dir/clean

CMakeFiles/sfm_node.dir/depend:
	cd /home/maotian/Sometest/sfm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/maotian/Sometest/sfm /home/maotian/Sometest/sfm /home/maotian/Sometest/sfm/build /home/maotian/Sometest/sfm/build /home/maotian/Sometest/sfm/build/CMakeFiles/sfm_node.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sfm_node.dir/depend

