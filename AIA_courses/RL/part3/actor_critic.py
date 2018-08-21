import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)

MAX_EPISODE = 500
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -60
RENDER = False
GAMMA = 0.95
LR_A = 0.0001
LR_C = 0.001

USE_ACTION_AS_INPUT = False

ENV_NAME = "Pendulum-v0"
env = gym.make(ENV_NAME)
env.seed(1)

n_obs = env.observation_space.shape[0]
is_discrete = hasattr(env.action_space, "n")

if is_discrete:
    n_action_space = env.action_space.n
else:
    n_action_space = env.action_space.shape[0]
print("n_obs:{0} n_action_space:{1}".format(n_obs, n_action_space))
print("discrete" if is_discrete else "continuous")

if not is_discrete:
    action_bound = env.action_space.high
    print("Action boundary: %s ~ %s" % (env.action_space.low[0], env.action_space.high[0]))

### Actor-Critic
class ActorCritic(object):
    def __init__(self, n_action_space, n_state, action_bound, sess):
        self.n_action_space = n_action_space
        self.n_state = n_state
        self.action_bound = action_bound
        self.sess = sess

        self.S = tf.placeholder(tf.float32, [1, self.n_state], 'state_input')
        self.gt_action = tf.placeholder(tf.float32, None, 'gt_action') #groundtruth action
        self.td_error = tf.placeholder(tf.float32, None, name='td_error')

        self.R = tf.placeholder(tf.float32, name='reward')
        self.v_ = tf.placeholder(tf.float32, [1, 1], 'next_exp_v')

        with tf.variable_scope('Actor'):
            #create actor network
            self.action, self.normal_dist = self.create_actor(self.S)

            with tf.name_scope('expected_value_for_actor'):
                #actor gradient = grad[logPi(a/s)*td_error]
                self.exp_v = self.normal_dist.log_prob(self.gt_action) * self.td_error
                self.exp_v = self.exp_v - 0.01 * self.normal_dist.entropy()

            with tf.name_scope('actor_train'):
                #wish to maximize Q value
                self.actor_train = tf.train.AdamOptimizer(learning_rate=LR_A).minimize(-self.exp_v)

        with tf.variable_scope('Critic'):
            #create critic network
            self.v = self.create_critic(self.S, self.gt_action)

            with tf.name_scope('squared_TD_error'):
                #critic gradient = grad[r + gamma * V(s_) - V(s)]
                critic_v_target = self.R + GAMMA * self.v_
                self.critic_td_error = tf.reduce_mean(critic_v_target - self.v)
                self.loss = tf.square(self.critic_td_error)
            
            with tf.name_scope('critic_train'):
                #wish to minimize the td error
                self.critic_train = tf.train.AdamOptimizer(learning_rate = LR_C).minimize(self.loss)
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.S: s}) #return a action with continuous real number

    def create_actor(self, s):
        l1 = tf.layers.dense(s, units=30, activation=tf.nn.relu, name='l1')
        mu = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.tanh, name='mu')
        sigma = tf.layers.dense(inputs=l1, units=1, activation=tf.nn.softplus, name='sigma')

        #game env 的 action boundary 是 -2 ~ 2, tanh() activation 出來 mu 要*2才會符合 action boundary
        #sigma + 1-e5 應該是不要讓分母變成0
        normal_dist = tf.distributions.Normal(tf.squeeze(mu*2), tf.squeeze(sigma + 1e-5))
        # action = tf.clip_by_value(normal_dist.sample(1), -self.action_bound[0], -self.action_bound[0], name='policy_action')
        action = tf.multiply(normal_dist.sample(1), self.action_bound, name='policy_action')
        return action, normal_dist

    def update_actor(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.S: s, self.gt_action: a, self.td_error: td}
        _, exp_v = self.sess.run([self.actor_train, self.exp_v], feed_dict)
        return exp_v

    def create_critic(self, s, a):
        l1 = tf.layers.dense(inputs=s, units=30, activation=tf.nn.relu, name='l1')
        if USE_ACTION_AS_INPUT:
            a = tf.reshape(a, (1, 1))
            l2 = tf.layers.dense(inputs=tf.concat([l1, a], axis=1), units=20, activation=tf.nn.relu, name='l2')
            v = tf.layers.dense(inputs=l2, units=1, activation=None, name='value')
        else:
            v = tf.layers.dense(inputs=l1, units=1, activation=None, name='value')
        return v

    def update_critic(self, s, r, s_, a=None):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        if a is None:
            v_ = self.sess.run(self.v, {self.S: s_})
            td_error, _ = self.sess.run([self.critic_td_error, self.critic_train],
                                            {self.S: s, self.v_: v_, self.R: r})
        else:
            v_ = self.sess.run(self.v, {self.S: s_, self.gt_action: a})
            td_error, _ = self.sess.run([self.critic_td_error, self.critic_train],
                                            {self.S: s, self.gt_action: a, self.v_: v_, self.R: r})
        return td_error

### Initial our Actor-Critic
tf.reset_default_graph()
sess = tf.InteractiveSession()
actor_critic = ActorCritic(n_action_space, n_obs, action_bound, sess)
sess.run(tf.global_variables_initializer())

### Main training
reward_history = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t, ep_rs = 0, 0
    while True:
        if RENDER:
            env.render()
        a = actor_critic.choose_action(s)
        s_, r, done, info = env.step(a)
        r /= 10 #scaled reward to converge faster
        if USE_ACTION_AS_INPUT:
            td_error = actor_critic.update_critic(s, r, s_, a)
        else:
            td_error = actor_critic.update_critic(s, r, s_) #critic gradient = grad[r + gamma * V(s_) - V(s)]
        actor_critic.update_actor(s, a, td_error) # actor gradient = grad[logPi(a|s)*td_error]

        s = s_
        t += 1
        ep_rs += r

        if t > MAX_EP_STEPS:
            reward_history.append(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs
            else:
                running_reward = running_reward * 0.9 + ep_rs * 0.1 # 期望是穩定的reward

            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True # rendering
            print("episode:",i_episode," reward:",int(running_reward))
            break

plt.plot(reward_history)

saver = tf.train.Saver()
save_path = saver.save(sess, "checkpoint/ac_v1.ckpt")

### Use checkpoint
saver.restore(sess, "checkpoint/ac_v1.ckpt")
s = env.reset()
i_step = 0
while True:
    env.render()
    a = actor_critic.choose_action(s)
    s_, r, done, info = env.step(a)
    s = s_
    i_step += 1
    if i_step > 300:
        break





