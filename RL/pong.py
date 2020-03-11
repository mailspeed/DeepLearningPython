import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("Pong-v0") # envoironment info
observation = env.reset()

for i in range(22):
	# fire ball after 20 frames
	if i > 20:
		plt.imshow(observation)
		plt.show()
	# get next observation
	observation, _, _, _ = env.step(1)

def preprocess_frame(frame):
	# remove the top of image and some background
	frame = frame[35:195, 10:150]
	# convert image to grayscale and shrink 1/2
	frame = frame[::2, ::2, 0]
	# set the background to 0
	frame[frame == 144] = 0
	frame[frame == 109] = 0
	# set ball and pat to 1
	frame[frame != 0] = 1
	return frame.astype(np.float).ravel()
	
obs_preprocessed = preprocess_frame(observation).reshape(80, 70)
plt.imshow(obs_preprocessed, cmap='gray')
plt.show()

observation_next, _, _, _ = env.step(1)
diff = preprocess_frame(observation_next) - preprocess_frame(observation)
plt.imshow(diff.reshape(80, 70), cmap='gray')
plt.show()

input_dim = 80*70
hidden_L1 = 400
hidden_L2 = 200
actions = [1, 2, 3]
n_actions = len(actions)
model = {}
with tf.variable_scope('L1', reuse=False):
	init_W1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(input_dim), dtype=tf.float32)
	model['W1'] = tf.get_variable("W1", [input_dim, hidden_L1], initializer=init_W1)

with tf.variable_scope('L2', reuse=False):
	init_W2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(hidden_L1), dtype=tf.float32)
	model['W2'] = tf.get_variable("W2", [hidden_L1, hidden_L2], initializer=init_W2)

def policy_forward(x):
	x = tf.matmul(x, model['W1'])
	x = tf.nn.relu(x)
	x = tf.matmul(x, model['W2'])
	p = tf.nn.softmax(x)
	return p
	
def discounted_rewards(reward, gamma):
	discounted_function = lambda a, v:a*gamma + v;
	reward_reverse = tf.scan(discounted_function, tf.reverse(reward, [True, False]))
	discounted_reward = tf.reverse(reward_reverse, [True, False])
	return discounted_reward
	
learning_rate = 0.001
gamma = 0.99
batch_size = 10

# define tensorflow placeholder
episode_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
episode_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])
episode_reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])

episode_discounted_reward = discounted_rewards(episode_reward, gamma)
episode_mean, episode_variance = tf.nn.moments(episode_discounted_reward, [0], shift=None)

# normalize discounted reward
episode_discounted_reward -= episode_mean
episode_discounted_reward /= tf.sqrt(episode_variance + 1e-6)

# optimizer setting
tf_aprob = policy_forward(episode_x)
loss = tf.nn.l2_loss(episode_y - tf_aprob)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss= episode_discounted_reward)
train_op = optimizer.apply_gradients(gradients)

# graphic initialization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train model saver setting
saver = tf.train.Saver(tf.global_variables())
save_path = 'checkpoints/pong_rl.ckpt'

obs_prev = None
xs, ys, rs, = [], [], []
reward_sum = 0
episode_number = 0
reward_window = None
reward_best = -22
history = []

observation = evn.reset()
while True:
	if True: env.render()
	# uncomment above to see agent
	# train and watch
	# preprocess observation and feed different images to network
	obs_cur = preprocess_frame(observation)
	obs_diff = obs_cur - obs_prev if obs_prev is not None else np.zeros(input_dim)
	obs_prev = obs_cur
	
	# sample one action with policy
	feed = {episode_x:np.reshape(obs_diff, (1, -1))};
	aprob = sess.run(tf_aprob, feed); aprob = aprob[0,:]
	action = np.random.choice(n_actions, p=aprob)
	label = np.zeros_like(aprob); label[action] = 1
	
	# return environment and get next observation, reward and status
	observation, reward, done, info = env.step(action+1)
	reward_sum += reward
	# record game history
	xs.append(obs_diff); ys.append(label);
	rs.append(reward)
	
	if done:
		history.append(reward_sum)
		reward_window = -21 if reward_window is None else np.mean(history[-100:])
		# update weight with stored values
		# (update policy)
		feed = {episode_x: np.vstack(xs), episode_y:np.vstack(ys), episode_reward:np.vstack(rs),}
		_ = sess.run(train_op, feed)
		print('episode {:2d}:reward:{:2.0f}'.format(episode_number, reward_sum))
		xs, ys, rs = [], [], []
		episode_number += 1
		observation = env.reset()
		reward_sum = 0
		# save best model after 10 scences
		if (episode_number % 10 == 0) & (reward_window > reward_best):
			saver.save(sess, save_path, global_step=episode_number)
			reward_best = reward_window
			print("Save best model {:2d}:{:2.5f} (reward window)".format(episode_number, reward_window))
			plt.plot(history)
			plt.show()
			
			observation = env.reset()
			while True:
				if True: env.render()
				obs_cur = preprocess_frame(observation)
				obs_diff = obs_cur - obs_prev if obs_prev is not None else np.zeros(input_dim)
				obs_prev = obs_cur
				
				feed = {tf_x: np.reshape(obs_diff, (1, -1))}
				aprob = sess.run(tf_aprob, feed); aprob= aprob[0, :]
				action = np.random.choice(n_actions, p=aprob)
				label = np.zeros_like(aprob);label[action]=1
				
				observation, reward, done, info = env.step(action+1)
				if done: observation=env.reset()
	
	
	
	