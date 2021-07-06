import numpy as np
import tensorflow as tf

class DQNAgent:
    """Class implementing DQN."""
    def __init__(self,
                 online_model,
                 target_model,
                 memory,
                 num_actions,
                 gamma,
                 target_update_freq,
                 update_target_params_ops,
                 batch_size,
                 learning_rate):
        self._online_model = online_model
        self._target_model = target_model
        self._memory = memory
        self._num_actions = num_actions
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._update_target_params_ops = update_target_params_ops
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._update_times = 0
        self._action1_ph_bz = tf.placeholder(tf.int32, [None, 2], name='action1_ph_bz')
        self._action2_ph_bz = tf.placeholder(tf.int32, [None, 2], name='action2_ph_bz')
        self._action1_ph_dh = tf.placeholder(tf.int32, [None, 2], name='action1_ph_dh')
        self._action2_ph_dh = tf.placeholder(tf.int32, [None, 2], name='action2_ph_dh')
        self._reward_ph_bz = tf.placeholder(tf.float32, name='reward_ph_bz')
        self._reward_ph_dh = tf.placeholder(tf.float32, name='reward_ph_dh')
        self._is_terminal_ph = tf.placeholder(tf.float32, name='is_terminal_ph')
        self._action1_chosen_by_online_ph_bz = tf.placeholder(tf.int32, [None, 2], name='action1_chosen_by_online_ph_bz')
        self._action2_chosen_by_online_ph_bz = tf.placeholder(tf.int32, [None, 2], name='action2_chosen_by_online_ph_bz')
        self._action1_chosen_by_online_ph_dh = tf.placeholder(tf.int32, [None, 2], name='action1_chosen_by_online_ph_dh')
        self._action2_chosen_by_online_ph_dh = tf.placeholder(tf.int32, [None, 2], name='action2_chosen_by_online_ph_dh')
        self._error_op_bz, self._train_op_bz,self._error_op_dh, self._train_op_dh =self._get_error_and_train_op(self._reward_ph_bz,
                                                                      self._reward_ph_dh,
                                                                      self._is_terminal_ph,
                                                                      self._action1_ph_bz,
                                                                      self._action2_ph_bz,
                                                                      self._action1_ph_dh,
                                                                      self._action2_ph_dh,
                                                                      self._action1_chosen_by_online_ph_bz,
                                                                      self._action2_chosen_by_online_ph_bz,
                                                                      self._action1_chosen_by_online_ph_dh,
                                                                      self._action2_chosen_by_online_ph_dh)
    def select_action(self, sess, state, model):
        """Select the action based on the current state."""
        feed_dict = {model['input_frames']: [state]}
        action1 = sess.run(model['action1'], feed_dict=feed_dict)
        action2 = sess.run(model['action2'], feed_dict=feed_dict)
        action1_bz = sess.run(model['action1_bz'], feed_dict=feed_dict)
        action2_bz = sess.run(model['action2_bz'], feed_dict=feed_dict)
        action1_dh = sess.run(model['action1_dh'], feed_dict=feed_dict)
        action2_dh = sess.run(model['action2_dh'], feed_dict=feed_dict)
        return action1, action2,action1_bz, action2_bz,action1_dh, action2_dh

    def get_mean_max_Q(self, sess, samples):
        mean_max1, mean_max2 = [], []
        INCREMENT = 10
        for i in range(0, len(samples), INCREMENT):
            feed_dict = {self._online_model['input_frames']:
                             samples[i: i + INCREMENT].astype(np.float32)}
            mean_max1.append(sess.run(self._online_model['mean_max_Q1'],
                                      feed_dict=feed_dict))
            mean_max2.append(sess.run(self._online_model['mean_max_Q2'],
                                      feed_dict=feed_dict))
        return np.mean(mean_max1), np.mean(mean_max2)

    def minimize_and_clip(self, optimizer, objective, total_n_streams):
        gradients = optimizer.compute_gradients(objective)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                common_net_coeff = 1.0 / total_n_streams
                if 'common' in var.name:
                    grad *= common_net_coeff
                    gradients[i] = (grad, var)

        return optimizer.apply_gradients(gradients)

    def _get_error_and_train_op(self,
                                reward_ph_bz,
                                reward_ph_dh,
                                is_terminal_ph,
                                action1_ph_bz,
                                action2_ph_bz,
                                action1_ph_dh,
                                action2_ph_dh,
                                action1_chosen_by_online_ph_bz,
                                action2_chosen_by_online_ph_bz,
                                action1_chosen_by_online_ph_dh,
                                action2_chosen_by_online_ph_dh):
        """Train the network"""

        Q_values_target1_bz = self._target_model['q_values1_bz']
        Q_values_online1_bz = self._online_model['q_values1_bz']
        Q_values_target2_bz = self._target_model['q_values2_bz']
        Q_values_online2_bz = self._online_model['q_values2_bz']
        Q_values_target1_dh = self._target_model['q_values1_dh']
        Q_values_online1_dh = self._online_model['q_values1_dh']
        Q_values_target2_dh = self._target_model['q_values2_dh']
        Q_values_online2_dh = self._online_model['q_values2_dh']

        max_q1_bz = tf.gather_nd(Q_values_target1_bz, action1_chosen_by_online_ph_bz)
        max_q2_bz = tf.gather_nd(Q_values_target2_bz, action2_chosen_by_online_ph_bz)
        max_q1_dh = tf.gather_nd(Q_values_target1_dh, action1_chosen_by_online_ph_dh)
        max_q2_dh = tf.gather_nd(Q_values_target2_dh, action2_chosen_by_online_ph_dh)

        target1_bz = reward_ph_bz + (1.0 - is_terminal_ph) * self._gamma * max_q1_bz
        target2_bz = reward_ph_bz + (1.0 - is_terminal_ph) * self._gamma * max_q2_bz
        target1_dh = reward_ph_dh + (1.0 - is_terminal_ph) * self._gamma * max_q1_dh
        target2_dh = reward_ph_dh + (1.0 - is_terminal_ph) * self._gamma * max_q2_dh
        gathered_outputs1_bz = tf.gather_nd(Q_values_online1_bz, action1_ph_bz, name='gathered_outputs1_bz')
        gathered_outputs2_bz = tf.gather_nd(Q_values_online2_bz, action2_ph_bz, name='gathered_outputs2_bz')
        gathered_outputs1_dh = tf.gather_nd(Q_values_online1_dh, action1_ph_dh, name='gathered_outputs1_dh')
        gathered_outputs2_dh = tf.gather_nd(Q_values_online2_dh, action2_ph_dh, name='gathered_outputs2_dh')

        loss1_bz = tf.reduce_mean(tf.square(target1_bz - gathered_outputs1_bz))
        loss2_bz = tf.reduce_mean(tf.square(target2_bz - gathered_outputs2_bz))
        loss3_bz = tf.reduce_mean(tf.square(gathered_outputs1_bz - gathered_outputs2_bz))
        loss_bz = loss1_bz * 0.4 + loss2_bz * 0.4 + loss3_bz * 0.2
        loss1_dh = tf.reduce_mean(tf.square(target1_dh - gathered_outputs1_dh))
        loss2_dh = tf.reduce_mean(tf.square(target2_dh - gathered_outputs2_dh))
        loss3_dh = tf.reduce_mean(tf.square(gathered_outputs1_dh - gathered_outputs2_dh))
        loss_dh = loss1_dh * 0.4 + loss2_dh * 0.4 + loss3_dh * 0.2

        train_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_op_bz = self.minimize_and_clip(train_optimizer, loss_bz, 2)
        train_op_dh = self.minimize_and_clip(train_optimizer, loss_dh, 2)

        error_op1_bz = tf.abs(gathered_outputs1_bz - target1_bz, name='abs_error1_bz')
        error_op2_bz = tf.abs(gathered_outputs2_bz - target2_bz, name='abs_error2_bz')
        error_op_bz = tf.reduce_mean(error_op1_bz * 0.5 + error_op2_bz * 0.5)
        error_op1_dh = tf.abs(gathered_outputs1_dh - target1_dh, name='abs_error1_dh')
        error_op2_dh = tf.abs(gathered_outputs2_dh - target2_dh, name='abs_error2_dh')
        error_op_dh = tf.reduce_mean(error_op1_dh * 0.5 + error_op2_dh * 0.5)

        return error_op_bz, train_op_bz,error_op_dh, train_op_dh

    def evaluate(self, sess, env):
        """Evaluate online model."""

        reward_total_bz = 0
        rewards_list_bz = []
        reward_total_dh = 0
        rewards_list_dh = []
        reward_total = 0
        rewards_list = []
        num_finished_episode = 0
        arrived_flag=[]
        env.Reset()

        while num_finished_episode < 5:
            old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal,is_arrived = env.GetState()
            next_action1, next_action2,next_action1_bz,next_action2_bz,next_action1_dh,next_action2_dh = self.select_action(sess, new_state, self._online_model)
            env.TakeAction(next_action1, next_action2,action1_bz,action2_bz,action1_dh,action2_dh)
            if not is_terminal:
                reward_total_bz += reward_bz
                reward_total_dh += reward_dh
            else:
                print(num_finished_episode)
                reward_total_bz += reward_bz
                reward_total_dh += reward_dh
                reward_total=reward_total_bz+reward_total_dh
                rewards_list.append(reward_total)
                arrived_flag.append(is_arrived)
                num_finished_episode += 1
                reward_total = 0
                reward_total_bz = 0
                reward_total_dh = 0

        return np.mean(rewards_list), np.std(rewards_list), np.max(rewards_list), np.min(rewards_list), rewards_list,arrived_flag

    def get_sample(self, env, sess):
        old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal,is_arrived = env.GetState()
        next_action1, next_action2,next_action1_bz,next_action2_bz,next_action1_dh,next_action2_dh = self.select_action(sess, new_state, self._online_model)
        env.TakeAction(next_action1, next_action2,next_action1_bz,next_action2_bz,next_action1_dh,next_action2_dh)

        return old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, float(is_terminal)

    def fit(self, sess, env, num_iterations, do_train=True):
        """Train using batch environment."""

        for t in range(num_iterations):
            # Prepare sample
            old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal = self.get_sample(env, sess)
            self._memory.append(old_state, action1, action2, action1_bz,action2_bz,action1_dh,action2_dh,reward_bz,reward_dh, new_state, is_terminal)

            # train.
            if do_train:
                old_state_list, action1_list, action2_list,action1_list_bz,action2_list_bz,action1_list_dh,action2_list_dh, reward_list_bz,reward_list_dh, new_state_list, is_terminal_list = self._memory.sample(self._batch_size)

                feed_dict = {self._target_model['input_frames']: new_state_list.astype(np.float32),
                             self._online_model['input_frames']: old_state_list.astype(np.float32),
                             self._action1_ph_bz: list(enumerate(action1_list_bz)),
                             self._action2_ph_bz: list(enumerate(action2_list_bz)),
                             self._action1_ph_dh: list(enumerate(action1_list_dh)),
                             self._action2_ph_dh: list(enumerate(action2_list_dh)),
                             self._reward_ph_bz: np.array(reward_list_bz).astype(np.float32),
                             self._reward_ph_dh: np.array(reward_list_dh).astype(np.float32),
                             self._is_terminal_ph: np.array(is_terminal_list).astype(np.float32),
                             }

                action1_chosen_by_online_bz, action2_chosen_by_online_bz,action1_chosen_by_online_dh, action2_chosen_by_online_dh = sess.run([self._online_model['action1_bz'], self._online_model['action2_bz'],self._online_model['action1_dh'], self._online_model['action2_dh']], feed_dict={self._online_model['input_frames']: new_state_list.astype(np.float32)})
                feed_dict[self._action1_chosen_by_online_ph_bz] = list(enumerate(action1_chosen_by_online_bz))
                feed_dict[self._action2_chosen_by_online_ph_bz] = list(enumerate(action2_chosen_by_online_bz))
                feed_dict[self._action1_chosen_by_online_ph_dh] = list(enumerate(action1_chosen_by_online_dh))
                feed_dict[self._action2_chosen_by_online_ph_dh] = list(enumerate(action2_chosen_by_online_dh))
#               action1_chosen_by_target, action2_chosen_by_target = sess.run([self._target_model['action1'], self._target_model['action2']], feed_dict={self._online_model['input_frames']: new_state_list.astype(np.float32)})
#               feed_dict[self._action1_chosen_by_target_ph] = list(enumerate(action1_chosen_by_target))
#               feed_dict[self._action2_chosen_by_target_ph] = list(enumerate(action2_chosen_by_target))
                sess.run(self._train_op_bz, feed_dict=feed_dict)
                sess.run(self._train_op_dh, feed_dict=feed_dict)
                print(self._update_times)
                self._update_times += 1

                if self._update_times % self._target_update_freq == 0:
                    sess.run(self._update_target_params_ops)
