"""
# mnist dqn agent 实现
# dqn 使用神经网络计算Q值（或者说用神经网络拟合Q函数）：
        https://pic3.zhimg.com/80/v2-0758fc770a22e5b838c73ae375b16132_1440w.jpg
# 参考资料：https://zhuanlan.zhihu.com/p/32818105
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from reinforcement_learning_mnist_classify.mnist_envirment import MnistEnviroment

layers = tf.keras.layers


class DqnAgent(object):
    def __init__(self):
        pass

    def build_net(self, input_height, input_width, channel, actions_num):
        """输入的图片经过CNN成2048维的特征向量，全连接层以特征向量为输入，输出10个Q-values，
        在这个状态所采取的动作a用onehot编码作为另一个输入，和这10个Q-values做点乘，
        显然除了该动作以外的其他值被置0，然后加起来即得到了这个Q(s,a)。
        这个值是由神经网络函数计算得来的估计值，
        需要用式(8)的Bellman最优化方程来计算长期回报的Q‘(s,a)，并用MSE来更新各权重。
        参考图片：https://pic1.zhimg.com/80/v2-51e63d6fc7fdbb9c9e84eb43110234cc_1440w.jpg
        :param input_height:
        :param input_width:
        :param channel:
        :param actions_num:
        :return:
        """
        img_input = layers.Input(shape=(input_height, input_width, channel),
                                 dtype='float32', name='image_inputs')

        # conv1
        conv1 = layers.Conv2D(32, 3, padding='same',
                              activation='relu', kernel_initializer='he_normal')(img_input)
        conv2 = layers.Conv2D(64, 3, strides=2, padding='same',
                              activation='relu', kernel_initializer='he_normal')(conv1)
        conv3 = layers.Conv2D(64, 3, strides=2, padding='same',
                              activation='relu', kernel_initializer='he_normal')(conv2)
        conv4 = layers.Conv2D(128, 3, strides=2, padding='same',
                              activation='relu', kernel_initializer='he_normal')(conv3)
        x = layers.Flatten()(conv4)
        x = layers.Dense(128, activation='relu')(x)
        # 通过神经网络计算出来的Q值，Q值最大值的索引为类别
        outputs_q = layers.Dense(actions_num, name='q_outputs')(x)

        # one hot input
        actions_input = layers.Input(shape=(actions_num, ), name='actions_input')
        q_value = layers.multiply([actions_input, outputs_q])
        # 得到batch_size个Q值[batch_size, 1]
        q_value = layers.Lambda(lambda x_input: tf.keras.backend.sum(x_input, axis=1, keepdims=True),
                                name='q_value')(q_value)

        model = tf.keras.models.Model(inputs=[img_input, actions_input], outputs=q_value)
        # 优化依然需要label, 只是使用dqn的思想
        model.compile(loss='mse', optimizer='adam')

        return model

    def copy_critic_to_actor(self, critic_model, actor_model):
        """
        :param critic_model: 用于训练的model（负责计算-估计）
        :param actor_model: 用于决策的model（负责动作-策略）
        :return:
        """
        critic_weights = critic_model.get_weights()
        actor_weights = actor_model.get_weights()
        for i in range(len(critic_weights)):
            # 将训练好的网络的参数copy到决策网络
            actor_weights[i] = critic_weights[i]

        actor_model.set_weights(actor_weights)

        return actor_model

    def get_q_values(self, model, state, dummy_actions):
        # 扩维的功能
        inputs = [state.reshape(1, *state.shape), dummy_actions]
        qvalues = model.predict(inputs)

        return qvalues

    def predict(self, model, states, num_actions):
        inputs = [states, np.ones(shape=(len(states), num_actions))]
        qvalues = model.predict(inputs)

        return np.argmax(qvalues, axis=1)

    def epsilon_calc(self, step, ep_min=0.01, ep_max=1,
                     ep_decay=0.0001, esp_total=1000):
        return max(ep_min, ep_max - (ep_max - ep_min) * step / esp_total)

    def epsilon_greedy(self, actor_q_model, env, state, step, dummy_actions,
                       ep_min=0.01, ep_decay=0.0001, ep_total=1000):
        epsilon = self.epsilon_calc(step, ep_min, 1, ep_decay, ep_total)
        if np.random.rand() < epsilon:
            return env.sample_actions(), 0
        qvalues = self.get_q_values(actor_q_model, state, dummy_actions)

        return np.argmax(qvalues), np.max(qvalues)

    def remember(self, memory, state, action, action_q, reward, next_state):
        memory.append([state, action, action_q, reward, next_state])

        return memory

    def sample_ram(self, sample_num, memory):
        return np.array(random.sample(memory, sample_num))

    def pre_remember(self, env, memory, pre_go=30):
        state = env.reset()
        for i in range(pre_go):
            rd_action = env.sample_actions()
            next_state, reward = env.step(rd_action)
            memory = self.remember(memory, state, rd_action, 0, reward, next_state)
            state = next_state

        return memory

    def replay(self, critic_model, memory, replay_size, num_actions, alpha, gamma):
        """经验回放:
        经验回放是强化学习玩atari游戏中不可思议的技巧之一。
        CNN本质上对数据的要求是独同分布，但是从电玩游戏环境过来的每一帧数据不管怎么处理实际上都是有关联的。
        为了打破这种关联性质，将每个时间步骤的经历 [公式] 也就是当前状态s，采取的动作a，得到的回报r以及下一个状态s'，
        当成记忆(memory)存储起来，而这个记忆是一个固定长度的double queue保证能忘记一些老的数据。
        在训练的时候，从memory随机采样，用记忆中的数据来计算更新DQN，这一步就满足了CNN的iid要求。
        虽然在逻辑上本项目不需要这一步，但是我还是将其加入到实现中以便以后在其他任务上扩展。
        为了实现方程(9)，我在经验中额外加入了当前的q值用于更新指数滑动平均。
        """
        if len(memory) < replay_size:
            return
        # 从记忆中i, i, d采样
        samples = self.sample_ram(replay_size, memory)
        # 展开所有样本的相关数据
        #这里next_states没用 因为和上一个state无关。
        states, actions, old_q, rewards, next_states = zip(*samples)
        states, actions, old_q, rewards = (np.array(states), np.array(actions).reshape(-1, 1),
                                           np.array(old_q).reshape(-1, 1),
                                           np.array(rewards).reshape(-1, 1))

        # 优化依然需要label
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_actions)
        # print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
        # 从actor获取下一个状态的q估计值 这里也没用 因为gamma=0 也就是不对bellman方程展开
        # inputs_ = [next_states,np.ones((replay_size,num_actions))]
        # qvalues = actor_q_model.predict(inputs_)

        # q = np.max(qvalues, axis=1, keepdims=True)
        q = 0
        # 应用Bellman方程对Q进行更新，将新的Q更新给critic（方程9）
        q_estimate = (1 - alpha) * old_q + alpha * (rewards.reshape(-1, 1) + gamma * q)
        # 训练估计模型
        history = critic_model.fit([states, actions_one_hot], q_estimate, epochs=1, verbose=0)

        return critic_model, np.mean(history.history['loss'])


def custom_load_mnist_data(file_path='mnist.npz'):
    """自定义数据读取器"""
    file_data = np.load(file_path)
    x_train, y_train = file_data['x_train'], file_data['y_train']
    x_test, y_test = file_data['x_test'], file_data['y_test']
    file_data.close()
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = custom_load_mnist_data()  # tf.keras.datasets.mnist.load_data()
    num_actions = len(set(y_test))
    image_w, image_h = x_train.shape[1:]

    x_train = x_train.reshape(*x_train.shape, 1)
    x_test = x_test.reshape(*x_test.shape, 1)
    # normalization
    x_train = x_train/255.0
    x_test = x_test/255.0

    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_actions)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_actions)

    test_image = x_train[0]
    plt.imshow(test_image.reshape(28, 28), 'gray')
    plt.show()
    plt.close()

    env = MnistEnviroment(x_train, y_train)
    from collections import deque
    import time
    memory = deque(maxlen=512)
    replay_size = 64
    epoches = 10
    pre_train_num = 256
    gamma = 0.  # every state is i.i.d
    alpha = 0.5
    forward = 512
    epislon_total = 2018

    agent_dqn = DqnAgent()
    # 用于决策
    actor_model = agent_dqn.build_net(image_h, image_w, channel=1, actions_num=num_actions)
    # 用于训练
    critic_model = agent_dqn.build_net(image_h, image_w, channel=1, actions_num=num_actions)
    actor_q_model = tf.keras.models.Model(inputs=actor_model.inputs,
                                          outputs=actor_model.get_layer('q_outputs').output)

    memory.clear()
    total_rewards = 0
    reward_rec = []
    # 填充初始经验池
    memory = agent_dqn.pre_remember(env, memory, pre_go=pre_train_num)
    every_copy_step = 128
    dummy_actions = np.ones((1, num_actions))

    pbar = tqdm(range(1, epoches+1))

    # 训练优化过程
    state = env.reset()
    for epoch in pbar:
        total_rewards = 0
        epoch_start = time.time()
        for step in range(forward):
            # 对每个状态使用epsilon_greedy选择
            action, q = agent_dqn.epsilon_greedy(actor_q_model, env, state, epoch,
                                                 dummy_actions=dummy_actions,
                                                 ep_min=0.01, ep_total=epislon_total)
            eps = agent_dqn.epsilon_calc(epoch, esp_total=epislon_total)
            # play
            next_state, reward = env.step(action)
            # 加入到经验记忆中
            memory = agent_dqn.remember(memory, state, action, q, reward, next_state)

            # 从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
            critic_model, loss = agent_dqn.replay(critic_model, memory, replay_size, num_actions, alpha, gamma)

            # 优化方式并不是同reward的期望来优化的
            total_rewards += reward
            state = next_state
            if step % every_copy_step == 0:
               actor_model = agent_dqn.copy_critic_to_actor(critic_model, actor_model)

        reward_rec.append(total_rewards)

        pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards, loss,
                                                                  int(time.time() - epoch_start), eps))

    critic_model.save('crtic_2000.HDF5')
    r5 = np.mean([reward_rec[i:i + 10] for i in range(0, len(reward_rec), 10)], axis=1)

    plt.plot(range(len(r5)), r5, c='b')
    plt.xlabel('iters')
    plt.ylabel('mean score')
    plt.show()
    plt.close()

    actor_model = agent_dqn.copy_critic_to_actor(critic_model, actor_model)

    # load trained model for the predict
    model_loaded = tf.keras.models.load_model('crtic_2000.HDF5')
    pred = agent_dqn.predict(actor_q_model, x_test, num_actions)

    from sklearn.metrics import accuracy_score
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))

    tf.keras.utils.plot_model(model_loaded, 'model.png', True)


