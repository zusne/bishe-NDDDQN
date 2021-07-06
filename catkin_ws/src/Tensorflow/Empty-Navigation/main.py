import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
# sys.path.append('/home/hitsz/.environments/drl_local_planner/lib/python3.5')
sys.path.append('/home/lsj/anaconda3/envs/BND/lib/python3.5')
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
# sys.path.remove('/home/hitsz/realsense_ws/devel/lib/python2.7/dist-packages')
print('-------------------main.py sys.path--------------------------------------------')
print(sys.path)
import tensorflow as tf
from GazeboWorld import GazeboWorld
from replayMemory import ReplayMemory
from model import create_duel_q_network, create_model
from agent import DQNAgent
import tensorflow as tf
import os 
#指定使用哪块GPU训练
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config=tf.compat.v1.ConfigProto()
#设置最大占有GPU不超过显存的70%
config.gpu_options.per_process_gpu_memory_fraction=0.8
#重点：设置动态分配GPU
config.gpu_options.allow_growth=True
TARGET_UPDATE_FREQENCY = 10
REPLAYMEMORY_SIZE = 30000
NUM_FIXED_SAMPLES = 50
NUM_BURN_IN = 50

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def get_fixed_samples(env, num_actions, num_samples):
    fixed_samples = []
    env.Reset()

    for _ in range(num_samples):
        old_state, action1, action2, reward, new_state, is_terminal,is_arrived= env.GetState()
        action1 = random.randrange(num_actions)
        action2 = random.randrange(num_actions)
        env.TakeAction(action1, action2)
        fixed_samples.append(new_state)
    return np.array(fixed_samples)

def main():
    parser = argparse.ArgumentParser(description='Train using Gazebo Simulations')
    parser.add_argument('--seed', default=10, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=(81,100), help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', default=0.1, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.00001, help='learning rate')
    parser.add_argument('--window_size', default=4, type = int, help=
                                'Number of frames to feed to the Q-network')
    parser.add_argument('--num_time', default=4, type=int, help='Number of steps in RNN')
    parser.add_argument('--num_actions', default=7, type=int, help='Number of actions')
    parser.add_argument('--batch_size', default=64, type = int, help=
                                'Batch size of the training part')
    parser.add_argument('--num_iteration', default=500000, type = int, help=
                                'number of iterations to train')
    parser.add_argument('--eval_every', default=0.01, type = float, help=
                                'What fraction of num_iteration to run between evaluations')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    batch_environment = GazeboWorld()
    print('Environment initialized')

    replay_memory = ReplayMemory(REPLAYMEMORY_SIZE, args.window_size, args.input_shape)
    online_model, online_params = create_model(args.window_size, args.input_shape, args.num_actions,
                    'online_model', create_duel_q_network, trainable=True)
    target_model, target_params = create_model(args.window_size, args.input_shape, args.num_actions,
                    'target_model', create_duel_q_network, trainable=False)
    update_target_params_ops = [t.assign(s) for s, t in zip(online_params, target_params)]
    reward1=[]
    agent = DQNAgent(online_model,
                    target_model,
                    replay_memory,
                    args.num_actions,
                    args.gamma,
                    TARGET_UPDATE_FREQENCY,
                    update_target_params_ops,
                    args.batch_size,
                    args.learning_rate)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sess.as_default():
        # saving and loading networks
        trainables = tf.trainable_variables()
        trainable_saver = tf.train.Saver(trainables, max_to_keep=1)
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        print('checkpoint:', checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            trainable_saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
         #make target_model equal to online_model
        sess.run(update_target_params_ops)

        print('Prepare fixed samples for mean max Q.')
        fixed_samples = get_fixed_samples(batch_environment, args.num_actions, NUM_FIXED_SAMPLES)
        print('length of fixed samples:')
        print(len(fixed_samples))
        # initialize replay buffer
        print('Burn in replay_memory.')
        agent.fit(sess, batch_environment, NUM_BURN_IN, do_train=False)

        # start training:
        fit_iteration = int(args.num_iteration * args.eval_every)
        filename="./picture/data.txt"
        filename1="./picture/arrvied.txt"
        for i in range(0, args.num_iteration, fit_iteration):
            # evaluate:
            reward_mean, reward_var, reward_max, reward_min, reward ,is_arrived= agent.evaluate(sess, batch_environment)
            mean_max_Q1, mean_max_Q2 = agent.get_mean_max_Q(sess, fixed_samples)
            print("%d, %f, %f, %f, %f, %f, %f"%(i, mean_max_Q1, mean_max_Q2, reward_mean, reward_var, reward_max, reward_min))
            # train:
            agent.fit(sess, batch_environment, fit_iteration, do_train=True)
            reward1=reward1+reward
            hit_times=np.arange(0,len(reward1),1)
            plt.ion()
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(hit_times,reward1)
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('reward')
            plt.savefig("./picture/reward/reward_%i"%i)
            plt.close()
            plt.ioff()
            with open(filename,"a") as file:
                for r in reward:
                    file.write(str(r)+","+"\n")
            with open(filename1,"a") as file:
                for ar in is_arrived:
                    file.write(str(ar)+","+"\n")
            trainable_saver.save(sess, 'saved_networks/', global_step=i)

        reward_mean, reward_var, reward_max, reward_min, reward = agent.evaluate(sess, batch_environment)
        mean_max_Q1, mean_max_Q2 = agent.get_mean_max_Q(sess, fixed_samples)
        print("%d, %f, %f, %f, %f, %f, %f" % (i, mean_max_Q1, mean_max_Q2, reward_mean, reward_var, reward_max, reward_min))

if __name__ == '__main__':
    main()
