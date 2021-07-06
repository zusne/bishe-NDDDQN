cmake_minimum_required(VERSION 2.8.3)
project(pioneer_utils)
set (CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs message_generation tf tf2 geometry_msgs sensor_msgs robot_state_publisher urdf)

catkin_package(
	
	CATKIN_DEPENDS roscpp rospy std_msgs message_runtime sensor_msgs tf tf2 robot_state_publisher urdf geometry_msgs sensor_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

#add_executable(teleop_p3at src/teleop_p3at.cpp)
#target_link_libraries(teleop_p3at ${catkin_LIBRARIES})

#add_executable(moving_alone src/moving_alone.cpp)
#target_link_libraries(moving_alone ${catkin_LIBRARIES})
