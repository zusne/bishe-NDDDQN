import sys
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
print('-------------------GazeboWorld.py sys.path1--------------------------------------------')
print(sys.path)
import tf
import rospy
import copy
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/home/lsj/anaconda3/envs/BND/lib/python3.5')
import cv2
import numpy as np

sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from preprocessor import HistoryPreprocessor
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, PointStamped
print('-------------------GazeboWorld.py sys.path2--------------------------------------------')
print(sys.path)

class GazeboWorld():
    def __init__(self, ns='',
                 start_location=(0, 0),
                 max_episode=1500,
                 window_size=4,
                 input_shape=(81, 100)):

        rospy.init_node('GazeboWorld', anonymous=False)

        # -----------Parameters-----------------------
        self.set_self_state = ModelState()
        self.set_self_state.model_name = ns + 'mobile_base'
        self.set_self_state.pose.position.x = start_location[0]
        self.set_self_state.pose.position.y = start_location[1]
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.set_self_state.twist.linear.x = 0.
        self.set_self_state.twist.linear.y = 0.
        self.set_self_state.twist.linear.z = 0.
        self.set_self_state.twist.angular.x = 0.
        self.set_self_state.twist.angular.y = 0.
        self.set_self_state.twist.angular.z = 0.
        self.set_self_state.reference_frame = 'world'
        self.goal_position = PointStamped()
        self.goal_position.header.stamp = rospy.get_rostime()
        self.goal_position.header.frame_id = "map"
        self.goal_position.point.x = -4.
        self.goal_position.point.y = 4.
        self.goal_position.point.z = 0.
        self.goal_table=[(-3.8,3.8)]
        self.old_distance=0.
        self.old_angle=0.
        self.min_distance=0.3
        self.max_distance=150.
        self.input_shape = input_shape
        self.bridge = CvBridge()
        self.object_state = [0, 0, 0, 0]
        self.object_name = []
        self.action1_table = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self.action2_table = [np.pi*45/180, np.pi*30/180, np.pi *
                              15/180, 0., -np.pi*15/180, -np.pi*30/180, -np.pi*45/180]
        self.self_speed = [0.7, 0.0]
        self.default_states = None
        self.start_table = [(0, 0)]
        self.depth_image = None
        self.bump = False

        self.time = 0
        self.max_episode = max_episode
        self.preprocessor = HistoryPreprocessor(
            self.input_shape, history_length=window_size)
        self.window_size = window_size
        self.state = {
            'old_state': np.zeros(shape=(input_shape[0], input_shape[1], window_size)),
            'action1': 0,
            'action2': 0,
            'reward': 0,
            'new_state': np.zeros(shape=(input_shape[0], input_shape[1], window_size)),
            'is_terminal': False,
            'is_arrived':False,
        }

        # -----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher(ns + 'cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            'gazebo/set_model_state', ModelState, queue_size=1)
        self.resized_depth_img = rospy.Publisher(
            ns + '/camera/depth/image_resized', Image, queue_size=1)
        self.goal_pub = rospy.Publisher('/goal_lsh', PointStamped, queue_size=1)
        self.object_state_sub = rospy.Subscriber(
            'gazebo/model_states', ModelStates, self.ModelStateCallBack)
        self.depth_image_sub = rospy.Subscriber(
            ns + '/camera/depth/image_raw', Image, self.DepthImageCallBack)
        self.odom_sub = rospy.Subscriber(
            ns + '/odom', Odometry, self.OdometryCallBack)
        self.bumper_sub = rospy.Subscriber(
            'bumper', ContactsState, self.BumperCallBack, queue_size=1)

        rospy.sleep(2.)
        rospy.on_shutdown(self.shutdown)

    def ModelStateCallBack(self, data):
        # self state
        # 把得到的状态数据转换成需要的状态，xy位置，z转角和三个速度
        idx = data.name.index(self.set_self_state.model_name)
        quaternion = (data.pose[idx].orientation.x,
                      data.pose[idx].orientation.y,
                      data.pose[idx].orientation.z,
                      data.pose[idx].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        self.self_state = [data.pose[idx].position.x,
                           data.pose[idx].position.y,
                           yaw,
                           data.twist[idx].linear.x,
                           data.twist[idx].linear.y,
                           data.twist[idx].angular.z]
        # 把gazebo中静态的物体的状态，转换成所需要的xy坐标以及z转角，和上面一样，现未使用，可在其他模块需要调用时使用
        for lp in range(len(self.object_name)):
            idx = data.name.index(self.object_name[lp])
            quaternion = (data.pose[idx].orientation.x,
                          data.pose[idx].orientation.y,
                          data.pose[idx].orientation.z,
                          data.pose[idx].orientation.w)
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]

            self.object_state[lp] = [data.pose[idx].position.x,
                                     data.pose[idx].position.y,
                                     yaw]

        if self.default_states is None:
            self.default_states = copy.deepcopy(data)

    def DepthImageCallBack(self, img):
        self.depth_image = img

    def OdometryCallBack(self, odometry):
        self.self_linear_x_speed = odometry.twist.twist.linear.x
        self.self_linear_y_speed = odometry.twist.twist.linear.y
        self.self_rotation_z_speed = odometry.twist.twist.angular.z

    def BumperCallBack(self, bumper_data):
        bump = False
        for state in bumper_data.states:
            if 'ground_plane' not in state.collision2_name:
                bump = True
                break

        self.bump = bump

    def GetDepthImageObservation(self):
        # ros image to cv2 image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.depth_image)  # 32FC1
        except Exception as e:
            raise e

        cv_img = np.array(cv_img, dtype=np.float32)
        # resize
        dim = (self.input_shape[1], self.input_shape[0]-1)
        cv_img = cv2.resize(
            cv_img, dim, interpolation=cv2.INTER_NEAREST)  # INTER_AREA
        cv_img[np.isnan(cv_img)] = 0.
        # normalize
        return(cv_img/5.)
    def GetPointObservation(self):
        robot_position=self.GetSelfState()
        point=np.zeros(shape=(1,self.input_shape[1]))
        point[0][self.input_shape[1]-4]=robot_position[0]
        point[0][self.input_shape[1]-3]=robot_position[1]
        point[0][self.input_shape[1]-2]=self.goal_position.point.x
        point[0][self.input_shape[1]-1]=self.goal_position.point.y
        return point
    def PublishDepthPrediction(self, depth_img):
        # cv2 image to ros image and publish
        cv_img = np.array(depth_img, dtype=np.float32)
        try:
            resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
        except Exception as e:
            raise e
        self.resized_depth_img.publish(resized_img)

    def GetSelfState(self):
        return self.self_state

    def GetSelfLinearXSpeed(self):
        return self.self_linear_x_speed

    def GetSelfOdomeSpeed(self):
        v = np.sqrt(self.self_linear_x_speed**2 + self.self_linear_y_speed**2)
        return [v, self.self_rotation_z_speed]

    def GetSelfSpeed(self):
        return np.array(self.self_speed)

    def GetBump(self):
        return self.bump

    def SetRobotPose(self):
        quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi*0.3, np.pi*0.3))
        start_location = self.start_table[np.random.randint(0, len(self.start_table))]
        object_state = copy.deepcopy(self.set_self_state)
        object_state.pose.orientation.x = quaternion[0]
        object_state.pose.orientation.y = quaternion[1]
        object_state.pose.orientation.z = quaternion[2]
        object_state.pose.orientation.w = quaternion[3]
        object_state.pose.position.x = start_location[0]
        object_state.pose.position.y = start_location[1]
        self.set_state.publish(object_state)
        rospy.sleep(0.1)

    def SetObjectPose(self):
        object_state = ModelState()
        state = copy.deepcopy(self.default_states)
        for i in range(len(self.default_states.name)):
            if 'mobile_base' not in state.name[i]:
                object_state.model_name = state.name[i]
                object_state.pose = state.pose[i]
                object_state.twist = state.twist[i]
                object_state.reference_frame = 'world'
                self.set_state.publish(object_state)
            rospy.sleep(0.1)
    def setGoal(self):
        #goal=self.goal_table[np.random.randint(0, len(self.goal_table))]
        #x=np.random.uniform(0, self.goal_table[0][0])
        #y=np.random.uniform(self.goal_table[0][1],0)
        self.goal_position.point.x = self.goal_table[0][0]
        self.goal_position.point.y = self.goal_table[0][1]
        '''print('new goal:')
        print(self.goal_position.point.x)
        print(self.goal_position.point.y)'''

    def ResetWorld(self):
        self.SetRobotPose()  # reset robot
        self.SetObjectPose()  # reset environment
        self.setGoal()
        rospy.sleep(0.1)

    def Control(self, action1, action2):
        self.self_speed[0] = self.action1_table[int(action1)]
        self.self_speed[1] = self.action2_table[int(action2)]
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)

    def shutdown(self):
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def GetRewardAndTerminate(self):
        terminate = False
        reset = False
        arrived=False
        [v, theta] = self.GetSelfOdomeSpeed()
        new_distance=self.get_distance()
        new_angle=self.get_relative_angle_to_goal(radian_or_degree='radian')/np.pi
        reward_distance=self.GetReward_Distance(new_distance,self.old_distance)
        reward_angle=self.GetReward_Angle(new_angle,self.old_angle)
        '''print('reward_distance2:')
        print(reward_distance)
        print('reward_angle:')
        print(reward_angle)
        print('v:')
        print(v)
        print('theta:')
        print(theta)
        print('reward_vw:')
        print(2*v * v * np.cos(2 * v * theta))'''
        self.old_distance=new_distance
        self.old_angle=new_angle
        reward =1*v * v * np.cos(2 * v * theta)+reward_distance+reward_angle
        if self.GetBump():
            reward = -10.
            terminate = True
            reset = True
        if self.Getgoal():
            reward = 200.
            terminate = True
            reset = True
            arrived=True
        if self.Getout():
            reward = -10.
            terminate = True
            reset = True
        if self.time > self.max_episode:
            reset = True

        return reward, terminate, reset,arrived
    def GetReward_Distance(self,new_distance,old_distance):
        reward_distance=0
        '''print('new_distance:')
        print(new_distance)
        print('old_distance:')
        print(old_distance)
        print('distance bigger?:')
        print(new_distance>old_distance)'''
        if new_distance<old_distance:
            reward_distance=0.1*(10-new_distance)
        elif new_distance>old_distance:
            reward_distance=-0.6
        else:
            reward_distance=0
        return reward_distance

    def GetReward_Angle(self,new_angle,old_angle):
        '''print('new_angle:')
        print(new_angle)
        print('old_angle:')
        print(old_angle)
        print('angle bigger?:')
        print(abs(new_angle)>abs(old_angle))'''
        reward_angle=0
        if abs(new_angle)<0.05:
            reward_angle=0.4
        elif abs(new_angle)<0.1:
            reward_angle=0.2
        elif abs(new_angle)<0.2:
            reward_angle=0.1
        elif abs(new_angle)<0.25:
            reward_angle=0
        elif abs(new_angle)<0.5:
            reward_angle=-0.2
        else:
            reward_angle=-0.4
        return reward_angle
    def GetState(self):
        return np.copy(self.state['old_state']), self.state['action1'], self.state['action2'], self.state['reward'], \
               np.copy(self.state['new_state']), self.state['is_terminal'],self.state['is_arrived']
    def TakeAction(self, action1, action2):
        robot_position=self.GetSelfState()
        with open("./picture/pointx.txt","a") as file:
            file.write(str(robot_position[0])+","+"\n")
        with open("./picture/pointy.txt","a") as file:
            file.write(str(robot_position[1])+","+"\n") 
        old_state = self.preprocessor.get_state()
        self.time += 1
        self.Control(action1, action2)
        rospy.sleep(0.1)
        state1 = self.GetDepthImageObservation()
        state2 = self.GetPointObservation()
        state  = np.concatenate([state1, state2], axis=0)
        reward, is_terminal, reset ,is_arrived= self.GetRewardAndTerminate()
        self.preprocessor.process_state_for_memory(state)
        new_state = self.preprocessor.get_state()
        #print(reward)
        self.state['old_state'] = old_state
        self.state['action1'] = action1
        self.state['action2'] = action2
        self.state['reward'] = reward
        self.state['new_state'] = new_state
        self.state['is_terminal'] = is_terminal
        self.state['is_arrived'] = is_arrived
        if reset:
            self.Reset()
    def Reset(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = 0.
        self.cmd_vel.publish(move_cmd)
        self.ResetWorld()
        self.preprocessor.reset()
        self.time = 0
        state1 = self.GetDepthImageObservation()
        state2 = self.GetPointObservation()
        state  = np.concatenate([state1, state2], axis=0)
        for _ in range(self.window_size):
            self.preprocessor.process_state_for_memory(state)
    def get_distance(self):
        # np.hypot(x, y) = np.sqrt(x**2 + y**2)  
        robot_position=self.GetSelfState()
        return np.hypot(robot_position[0] - self.goal_position.point.x, robot_position[1] - self.goal_position.point.y)
    def get_relative_angle_to_goal(self, radian_or_degree):
        robot_position=self.GetSelfState()
        angle_radian = np.arctan2(self.goal_position.point.y - robot_position[1],
                                  self.goal_position.point.x - robot_position[0]) - robot_position[2]
        angle = angle_radian
        if angle_radian / np.pi < -1:
            angle = angle_radian + 2 * np.pi
        elif angle_radian / np.pi > 1:
            angle = angle_radian - 2 * np.pi
        if radian_or_degree == "radian":
            return angle
        elif radian_or_degree == "degree":
            angle_degree = 180 * angle / np.pi
            return angle_degree

    def Getgoal(self):
        distance=self.get_distance()
        if distance<self.min_distance:
            return True
        else:
            return False
    def Getout(self):
        distance=self.get_distance()
        if distance>self.max_distance:
            return True
        else:
            return False
    def __publish_goal(self, x, y, z):
        """
        Publishing goal (x, y, theta)
        :param x x-position of the goal
        :param y y-position of the goal
        :param theta theta-position of the goal
        """
        goal = PointStamped()
        goal.header.stamp = rospy.get_rostime()
        goal.header.frame_id = "map"
        goal.point.x = x
        goal.point.y = y
        goal.point.z = z
        self.goal_pub.publish(goal)
