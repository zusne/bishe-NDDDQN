# CMake generated Testfile for 
# Source directory: /home/lsj/catkin_ws/src/geometry2/tf2_sensor_msgs
# Build directory: /home/lsj/catkin_ws/build/geometry2/tf2_sensor_msgs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_tf2_sensor_msgs_nosetests_test.test_tf2_sensor_msgs.py "/home/lsj/catkin_ws/build/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/melodic/share/catkin/cmake/test/run_tests.py" "/home/lsj/catkin_ws/build/test_results/tf2_sensor_msgs/nosetests-test.test_tf2_sensor_msgs.py.xml" "--return-code" "\"/usr/bin/cmake\" -E make_directory /home/lsj/catkin_ws/build/test_results/tf2_sensor_msgs" "/home/lsj/anaconda3/bin/nosetests -P --process-timeout=60 /home/lsj/catkin_ws/src/geometry2/tf2_sensor_msgs/test/test_tf2_sensor_msgs.py --with-xunit --xunit-file=/home/lsj/catkin_ws/build/test_results/tf2_sensor_msgs/nosetests-test.test_tf2_sensor_msgs.py.xml")
add_test(_ctest_tf2_sensor_msgs_rostest_test_test.launch "/home/lsj/catkin_ws/build/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/melodic/share/catkin/cmake/test/run_tests.py" "/home/lsj/catkin_ws/build/test_results/tf2_sensor_msgs/rostest-test_test.xml" "--return-code" "/usr/bin/python3 /opt/ros/melodic/share/rostest/cmake/../../../bin/rostest --pkgdir=/home/lsj/catkin_ws/src/geometry2/tf2_sensor_msgs --package=tf2_sensor_msgs --results-filename test_test.xml --results-base-dir \"/home/lsj/catkin_ws/build/test_results\" /home/lsj/catkin_ws/src/geometry2/tf2_sensor_msgs/test/test.launch ")
