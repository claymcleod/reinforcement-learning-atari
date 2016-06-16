# TODO(clay): moves global frame count (used to calculate epsilon) to tensorflow from session restore
# TODO(clay): implement prioritized memory replay
# TODO(clay): implement dueling Q-networks
# TODO(clay): integrate flags

import os
import gym
import tflearn
import numpy as np
import tensorflow as tf

from dimensions import dimensions_for_env
from random import random, randint
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

# Extended data table 1, appendix
tf.app.flags.DEFINE_integer('minibatch_size', 32, 'Number of training cases over which gradient descent (SGD) update is computed.')
tf.app.flags.DEFINE_integer('replay_memory_size', 100, 'SGD updates are sampled from this number of most recent frames.')
tf.app.flags.DEFINE_integer('agent_history_length', 4, 'The number of most recent frames experienced by the agent that are given as input to the Q network.')
tf.app.flags.DEFINE_integer('target_network_update_frequency', 100, 'The frequency (measured in the number of parameter updates) with which the target network is updated (this corresponds to the parameter C from Algorithm 1.)')
tf.app.flags.DEFINE_float('discount_factor', 0.99, 'Discount factor gamma used in the Q-learning update.')
tf.app.flags.DEFINE_integer('action_repeat', 4, 'Repeat each acton selected by the agent this many times. Using a value of 4 results in the agent seeing only every 4th input frame.')
tf.app.flags.DEFINE_integer('update_frequency', 4, 'The number of actions selected by the agent between successive SGD updates. Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates.')
tf.app.flags.DEFINE_float('learning_rate', 0.000025, 'The learning rate used by RMSProp.')
tf.app.flags.DEFINE_float('gradient_momentum', 0.95, 'Gradient momentum used by RMSProp.')
tf.app.flags.DEFINE_float('squared_gradient_momentum', 0.95, 'Squared gradient (denominator) momentum used by RMSProp.')
tf.app.flags.DEFINE_float('min_squared_gradient', 0.01, 'Constant added to the squared gradient in the denominator of the RMSProp update.')
tf.app.flags.DEFINE_float('initial_exploration', 0.1, 'Initial value of the epsilon in the epsilon-greedy exploration.')
tf.app.flags.DEFINE_float('final_exploration', 0.1, 'Final value of the epsilon in the epsilon-greedy exploration.')
tf.app.flags.DEFINE_integer('final_exploration_frame', 100000, 'The number of frames over which the initial value of epsilon is linearly annealed to its final value.')
tf.app.flags.DEFINE_integer('replay_start_size', 50, 'A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory.')
tf.app.flags.DEFINE_integer('no_op_max', 30, 'Maximum number of "do nothing" actions to be performed by the agent at the start of an epsiode.')

FLAGS = tf.app.flags.FLAGS

class DQN(object):

    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.num_actions = self.env.action_space.n
        self.state_buffer = deque()
        self.dimensions = dimensions_for_env(env_name)

    def _get_epsilon(self, iteration):
        """Linearly interpolate learning rate based on parameters"""

        alpha = iteration / FLAGS.final_exploration_frame
        if alpha > 1: alpha = 1
        return (1. - alpha) * FLAGS.initial_exploration + (alpha) * FLAGS.final_exploration

    def _build_Q_network(self):
        trainable_params_start = len(tf.trainable_variables())
        inputs = tf.placeholder(tf.float32, [None, FLAGS.agent_history_length, self.dimensions["network_in_y"], self.dimensions["network_in_x"]])
        transposed_input = tf.transpose(inputs, [0, 2, 3, 1])
        conv1 = tflearn.conv_2d(transposed_input, 32, 8, strides=4, activation='relu')
        conv2 = tflearn.conv_2d(conv1, 64, 4, strides=2, activation='relu')
        flatten = tflearn.fully_connected(conv2, 256, activation='relu')
        softmax = tflearn.fully_connected(flatten, self.num_actions)
        argmax = tf.argmax(softmax, dimension=1)
        return inputs, softmax, tf.trainable_variables()[trainable_params_start:], argmax

    def get_q_for_state(self, state):
        assert self.session != None, "Must initialize the DQN using .setup()!"
        return self.session.run([self.q_values], feed_dict={self.q_inputs: [state]})[0]

    def get_action_for_state(self, state):
        assert self.session != None, "Must initialize the DQN using .setup()!"
        return self.session.run(self.q_action, feed_dict={self.q_input: [state]})[0]

    def _build_training_graph(self):
        self.r_t = tf.placeholder(tf.float32)
        self.a_t_selected = tf.placeholder(tf.float32, [None, self.num_actions])
        self.q_input, self.q_values, self.q_params, self.q_action = self._build_Q_network()
        self.qt_input, self.qt_values, self.qt_params, _ = self._build_Q_network()
        self.update_target_params = [self.qt_params[i].assign(self.q_params[i]) \
                                     for i in range(len(self.qt_params))]
        self.actor_fn = self.r_t + tf.mul(FLAGS.discount_factor, self.qt_values)
        selected_a_q = tf.reduce_sum(tf.mul(self.q_values, self.a_t_selected), reduction_indices=1)
        selected_a_actor = tf.reduce_sum(tf.mul(self.actor_fn, self.a_t_selected), reduction_indices=1)
        self.cost_fn = tflearn.mean_square(selected_a_q, selected_a_actor)
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate, momentum=FLAGS.gradient_momentum)
        self.grad_fn = optimizer.minimize(self.cost_fn, var_list=self.q_params)

    def _get_action(self, s_t, iteration, no_ops):
        action = 0

        if s_t is None or random() < self._get_epsilon(iteration):
            action = randint(0, self.num_actions-1)
        else:
            action = self.get_action_for_state(s_t)

        while no_ops > FLAGS.no_op_max and action == 0:
            print("Max no_ops!")
            if s_t is None or random() < self._get_epsilon(iteration):
                action = randint(0, self.num_actions-1)
            else:
                action = self.get_action_for_state(s_t)

        return action

    def _preprocess_img(self, observation_t):
        """Borrowed from https://github.com/tflearn/tflearn/blob/master/examples/reinforcement_learning/atari_1step_qlearning.py"""
        return resize(rgb2gray(observation_t), self.dimensions["resize_to"])[self.dimensions["start"]:self.dimensions["stop"], :]

    def _step(self, action):
        s_t = None
        if len(self.state_buffer):
            s_t = np.array(self.state_buffer)

        observation_t, reward_t, done, _ = self.env.step(action)
        phi_observation_t = self._preprocess_img(observation_t)

        while len(self.state_buffer) >= FLAGS.agent_history_length:
            self.state_buffer.popleft() # pop

        while len(self.state_buffer) < FLAGS.agent_history_length:
            self.state_buffer.append(phi_observation_t)

        s_t_plus_one = np.array(self.state_buffer)

        return s_t, reward_t, s_t_plus_one, done

    def _get_cost(self, s_t, r_t, s_t1, a_t_selected):
        return self.cost_fn.eval(session=self.session, feed_dict={
            self.q_input: [s_t],
            self.r_t: r_t,
            self.qt_input: [s_t1],
            self.a_t_selected: [a_t_selected]
        })

    def _update_q_gradients(self, s_t, r_t, s_t1, a_t_selected):
        self.grad_fn.run(session=self.session, feed_dict={
            self.q_input: [s_t],
            self.r_t: r_t,
            self.qt_input: [s_t1],
            self.a_t_selected: [a_t_selected]
        })

    def _update_target_network(self):
        self.session.run(self.update_target_params)

    def _get_initial_state(self):
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self._preprocess_img(x_t)
        s_t = np.stack([x_t for i in range(FLAGS.agent_history_length)], axis=0)

        for i in range(FLAGS.agent_history_length):
            self.state_buffer.append(x_t)
        return s_t

    def _add_memory(self, s_t, r_t, s_t1, a_t_selected):
        if len(self.replay_memory) >= FLAGS.replay_memory_size:
            self.replay_memory.popleft()

        self.replay_memory.append((s_t, r_t, s_t1, a_t_selected))

    def _perform_memory_replay(self):
        s_t_all = []
        r_t_all = []
        s_t1_all = []
        a_t_selected_all = []

        for i in range(FLAGS.minibatch_size):
            replay_index = randint(0, len(self.replay_memory)-1) # Never pick last memory to simplify math
            (s_t, r_t, s_t1, a_t_selected) = self.replay_memory[replay_index]
            s_t_all.append(s_t)
            r_t_all.append(r_t)
            s_t1_all.append(s_t1)
            a_t_selected_all.append(a_t_selected)

        self._update_q_gradients(np.array(s_t_all)[0], np.array(r_t_all)[0], np.array(s_t1_all)[0], np.array(a_t_selected_all)[0])

    def _new_episode(self, frames, verbose=True, render=True):
        s_t = self._get_initial_state()
        done = False
        iteration = frames
        cost = 0.0
        score = 0.0
        a_t = 0
        no_ops = 0

        while not done:
            if iteration % FLAGS.action_repeat == 0:
                a_t = self._get_action(s_t, iteration, no_ops)
                if a_t == 0:
                    no_ops += 1
                else:
                    no_ops = 0

            s_t, r_t, s_t1, done = self._step(a_t)

            a_t_selected = np.zeros([self.num_actions])
            a_t_selected[a_t] = 1

            if verbose:
                cost += self._get_cost(s_t, r_t, s_t1, a_t_selected)

            if render:
                self.env.render()

            score += r_t
            iteration += 1
            self._add_memory(s_t, r_t, s_t1, a_t_selected)
            if iteration > FLAGS.replay_start_size:
                self._perform_memory_replay()
                if iteration % FLAGS.target_network_update_frequency == 0:
                    self._update_target_network()

        avg_cost = (cost / iteration)

        return avg_cost, score, iteration

    def setup(self):
        self._build_Q_network()
        self._build_training_graph()
        if os.environ["TF_THREADS"]:
            print("Using {} threads...".format(os.environ["TF_THREADS"]))
            self.session = tf.Session(config=tf.ConfigProto(
                           intra_op_parallelism_threads=int(os.environ["TF_THREADS"])))
        else:
            self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, save_model_every=1, verbose=True, render=True):
        self.setup()
        self.replay_memory = deque()

        saver = tf.train.Saver()
        save_path = './{0}.cpkl'.format(self.env_name)

        try:
            saver.restore(self.session, save_path)
            print("Restoring from previous session...")
        except:
            print("Starting from scratch...")

        num_episodes = 0
        frames = 0

        while True:
            avg_cost, score, frames = self._new_episode(frames, verbose=verbose, render=render)
            output = "#%d" % (num_episodes)
            if verbose:
                output = "#%d | Avg cost: %0.2f | Score: %d | Espilon: %0.3f | Replay Size: %d" % (num_episodes, avg_cost, score, self._get_epsilon(frames), len(self.replay_memory))
            print(output)
            if num_episodes % save_model_every == 0:
                saver.save(self.session, save_path)
            num_episodes += 1
