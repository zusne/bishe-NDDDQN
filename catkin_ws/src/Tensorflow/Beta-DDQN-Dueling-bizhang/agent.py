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
        self._l = []
        self.beta_constant=0.001
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
        self._action1_ph = tf.placeholder(tf.int32, [None, 2], name='action1_ph')
        self._action2_ph = tf.placeholder(tf.int32, [None, 2], name='action2_ph')
        self._reward_ph = tf.placeholder(tf.float32, name='reward_ph')
        self._is_terminal_ph = tf.placeholder(tf.float32, name='is_terminal_ph')
        self._action1_chosen_by_online_ph = tf.placeholder(tf.int32, [None, 2], name='action1_chosen_by_online_ph')
        self._action2_chosen_by_online_ph = tf.placeholder(tf.int32, [None, 2], name='action2_chosen_by_online_ph')
#DQN:
#       self._action1_chosen_by_target_ph = tf.placeholder(tf.int32, [None, 2], name='action1_chosen_by_target_ph')
#       self._action2_chosen_by_target_ph = tf.placeholder(tf.int32, [None, 2], name='action2_chosen_by_target_ph')
        self._error_op, self._train_op ,self._loss_op= self._get_error_and_train_op(self._reward_ph,
                                                                      self._is_terminal_ph,
                                                                      self._action1_ph,
                                                                      self._action2_ph,
                                                                      self._action1_chosen_by_online_ph,
                                                                      self._action2_chosen_by_online_ph)
    def data_return(self):
        return self._l
    def increase_beta(self):
        self.beta_constant=self.beta_constant+0.001
    def select_action(self, env,sess, state, model):
        """Select the action based on the current state."""
        last_action2=env.get_last_action2()
        feed_dict = {model['input_frames']: [state]}
        action1 = sess.run(model['action1'], feed_dict=feed_dict)
        action2 = sess.run(model['action2'], feed_dict=feed_dict)
        q_network2=model['q_values2']
        q_network_e2=tf.exp(q_network2)
        q_1=q_network2[0][last_action2]
        q_2=tf.reduce_max(q_network2, axis=[1])
        q_3=tf.reduce_sum(q_network_e2, axis=[1])
        beta=(q_1-q_2)/q_3
        beta1=sess.run(beta,feed_dict=feed_dict)
        if beta1<self.beta_constant:
            action2=last_action2
        return action1, action2

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
                                reward_ph,
                                is_terminal_ph,
                                action1_ph,
                                action2_ph,
                                action1_chosen_by_online_ph,
                                action2_chosen_by_online_ph):
        """Train the network"""

        Q_values_target1 = self._target_model['q_values1']
        Q_values_online1 = self._online_model['q_values1']
        Q_values_target2 = self._target_model['q_values2']
        Q_values_online2 = self._online_model['q_values2']

        max_q1 = tf.gather_nd(Q_values_target1, action1_chosen_by_online_ph)
        max_q2 = tf.gather_nd(Q_values_target2, action2_chosen_by_online_ph)
#DQN：
#       max_q1 = tf.gather_nd(Q_values_target1, action1_chosen_by_target_ph)
#       max_q2 = tf.gather_nd(Q_values_target2, action2_chosen_by_target_ph)
        target1 = reward_ph + (1.0 - is_terminal_ph) * self._gamma * max_q1
        target2 = reward_ph + (1.0 - is_terminal_ph) * self._gamma * max_q2
        gathered_outputs1 = tf.gather_nd(Q_values_online1, action1_ph, name='gathered_outputs1')
        gathered_outputs2 = tf.gather_nd(Q_values_online2, action2_ph, name='gathered_outputs2')

        loss1 = tf.reduce_mean(tf.square(target1 - gathered_outputs1))
        loss2 = tf.reduce_mean(tf.square(target2 - gathered_outputs2))
        loss3 = tf.reduce_mean(tf.square(gathered_outputs1 - gathered_outputs2))
        loss = loss1 * 0.4 + loss2 * 0.4 + loss3 * 0.2

        train_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_op = self.minimize_and_clip(train_optimizer, loss, 2)
        error_op1 = tf.abs(gathered_outputs1 - target1, name='abs_error1')
        error_op2 = tf.abs(gathered_outputs2 - target2, name='abs_error2')
        error_op = tf.reduce_mean(error_op1 * 0.5 + error_op2 * 0.5)

        return error_op, train_op, loss

    def evaluate(self, sess, env):
        """Evaluate online model."""

        reward_total = 0
        rewards_list = []
        num_finished_episode = 0
        env.Reset()

        while num_finished_episode < 5:
            old_state, action1, action2, reward, new_state, is_terminal = env.GetState()
            next_action1, next_action2 = self.select_action(env,sess,new_state, self._online_model)
            env.TakeAction(next_action1, next_action2)
            if not is_terminal:
                reward_total += reward
            else:
                print(num_finished_episode)
                reward_total += reward
                if reward_total > -9:
                    rewards_list.append(reward_total)
                    num_finished_episode += 1
                reward_total = 0

        return np.mean(rewards_list), np.std(rewards_list), np.max(rewards_list), np.min(rewards_list), rewards_list

    def get_sample(self, env, sess):
        old_state, action1, action2, reward, new_state, is_terminal = env.GetState()
        next_action1, next_action2 = self.select_action(env,sess,new_state, self._online_model)
        env.TakeAction(next_action1, next_action2)

        return old_state, action1, action2, reward, new_state, float(is_terminal)

    def fit(self, sess, env, num_iterations, do_train=True):
        """Train using batch environment."""

        for t in range(num_iterations):
            # Prepare sample
            old_state, action1, action2, reward, new_state, is_terminal = self.get_sample(env, sess)
            self._memory.append(old_state, action1, action2, reward, new_state, is_terminal)

            # train.
            if do_train:
                old_state_list, action1_list, action2_list, reward_list, new_state_list, is_terminal_list = self._memory.sample(self._batch_size)

                feed_dict = {self._target_model['input_frames']: new_state_list.astype(np.float32),
                             self._online_model['input_frames']: old_state_list.astype(np.float32),
                             self._action1_ph: list(enumerate(action1_list)),
                             self._action2_ph: list(enumerate(action2_list)),
                             self._reward_ph: np.array(reward_list).astype(np.float32),
                             self._is_terminal_ph: np.array(is_terminal_list).astype(np.float32),
                             }

                action1_chosen_by_online, action2_chosen_by_online = sess.run([self._online_model['action1'], self._online_model['action2']], feed_dict={self._online_model['input_frames']: new_state_list.astype(np.float32)})
                feed_dict[self._action1_chosen_by_online_ph] = list(enumerate(action1_chosen_by_online))
                feed_dict[self._action2_chosen_by_online_ph] = list(enumerate(action2_chosen_by_online))
#               action1_chosen_by_target, action2_chosen_by_target = sess.run([self._target_model['action1'], self._target_model['action2']], feed_dict={self._online_model['input_frames']: new_state_list.astype(np.float32)})
#               feed_dict[self._action1_chosen_by_target_ph] = list(enumerate(action1_chosen_by_target))
#               feed_dict[self._action2_chosen_by_target_ph] = list(enumerate(action2_chosen_by_target))
                sess.run(self._train_op, feed_dict=feed_dict)
                #self._e.append(sess.run(tf.reduce_mean(sess.run(self._error_op, feed_dict=feed_dict))))
                self._l.append(sess.run(self._loss_op, feed_dict=feed_dict))
                print(self._update_times)
                self._update_times += 1

                if self._update_times % self._target_update_freq == 0:
                    sess.run(self._update_target_params_ops)
