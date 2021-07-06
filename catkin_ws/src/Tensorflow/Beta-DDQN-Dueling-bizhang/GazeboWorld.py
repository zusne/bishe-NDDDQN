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
                 input_shape=(80, 100)):

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
        self.input_shape = input_shape
        self.bridge = CvBridge()
        self.object_state = [0, 0, 0, 0]
        self.object_name = []
        self.action1_table = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        self.action2_table = [np.pi*45/180, np.pi*30/180, np.pi *
                              15/180, 0., -np.pi*15/180, -np.pi*30/180, -np.pi*45/180]
        self.self_speed = [0.7, 0.0]
        self.last_action2=0
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
            'is_terminal': False
        }

        # -----------Publisher and Subscriber-------------
        self.cmd_vel = rospy.Publisher(ns + 'cmd_vel', Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            'gazebo/set_model_state', ModelState, queue_size=1)
        self.resized_depth_img = rospy.Publisher(
            ns + '/camera/depth/image_resized', Image, queue_size=1)
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
        dim = (self.input_shape[1], self.input_shape[0])
        cv_img = cv2.resize(
            cv_img, dim, interpolation=cv2.INTER_NEAREST)  # INTER_AREA
        cv_img[np.isnan(cv_img)] = 0.
        # normalize
        return(cv_img/5.)

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
        quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
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
    def ResetWorld(self):
        self.SetRobotPose()  # reset robot
        self.SetObjectPose()  # reset environment
        rospy.sleep(0.1)
    def get_last_action2(self):
        return self.last_action2

    def Control(self, action1, action2):
        self.last_action2=action2
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
        [v, theta] = self.GetSelfOdomeSpeed()
        reward = 2*v * v * np.cos(2 * v * theta)-0.1
        if self.GetBump():
            reward = -10.
            terminate = True
            reset = True
        if self.time > self.max_episode:
            reset = True

        return reward, terminate, reset


    def GetState(self):
        return np.copy(self.state['old_state']), self.state['action1'], self.state['action2'], self.state['reward'], \
               np.copy(self.state['new_state']), self.state['is_terminal']

    def TakeAction(self, action1, action2):
        robot_position=self.GetSelfState()
        '''with open("./picture/pointx.txt","a") as file:
            file.write(str(robot_position[0])+","+"\n")
        with open("./picture/pointy.txt","a") as file:
            file.write(str(robot_position[1])+","+"\n")'''     
        old_state = self.preprocessor.get_state()
        self.time += 1
        self.Control(action1, action2)
        rospy.sleep(0.1)
        state = self.GetDepthImageObservation()
        reward, is_terminal, reset = self.GetRewardAndTerminate()
        self.preprocessor.process_state_for_memory(state)
        new_state = self.preprocessor.get_state()
        with open("./picture/reward_now.txt","a") as file:
            file.write(str(reward)+","+"\n")
        self.state['old_state'] = old_state
        self.state['action1'] = action1
        self.state['action2'] = action2
        self.state['reward'] = reward
        self.state['new_state'] = new_state
        self.state['is_terminal'] = is_terminal
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
        state = self.GetDepthImageObservation()
        for _ in range(self.window_size):
            self.preprocessor.process_state_for_memory(state)



