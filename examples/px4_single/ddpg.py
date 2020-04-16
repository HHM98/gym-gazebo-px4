import random
import time

import numpy as np
import tensorflow as tf

from keras import Sequential, optimizers
from keras.layers import Dense, Input, Lambda, concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as k_b
import memory



class DDPG:
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart, maxMargin):
        self.sess = k_b.get_session()

        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discount_factor = discountFactor
        self.learning_rate = learningRate
        self.learn_start = learnStart
        self.max_margin = maxMargin

    def init_net_works(self, hidden_layers):
        self.actor = self.create_actor_model(self.input_size, self.output_size, hidden_layers, "selu",
                                             self.learning_rate)
        self.actor_target = self.create_actor_model(self.input_size, self.output_size, hidden_layers, "selu",
                                                    self.learning_rate)
        self.actor_target.set_weights(self.actor.get_weights())

        # create critic
        self.critic = self.create_critic_model(self.input_size, hidden_layers, "selu", self.learning_rate)
        self.critic_target = self.create_critic_model(self.input_size, hidden_layers, "selu", self.learning_rate)
        self.critic_target.set_weights(self.critic.get_weights())

        self.get_critic_grad = self.critic_gradient()
        self.actor_optimizer()

    def create_actor_model(self, input_size, output_size, hidden_layers, activation_type, learning_rate):
        inputs = Input(shape=(input_size,), name='state_input')
        x = Dense(hidden_layers[0], activation=activation_type)(inputs)

        for idx in range(1, len(hidden_layers)):
            x = Dense(hidden_layers[idx], activation=activation_type)(x)

        x = Dense(output_size, activation='tanh')(x)
        outputs = Lambda(lambda x: x * self.max_margin)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        return model

    def create_critic_model(self, input_size, hidden_layers, activation_type, learning_rate):
        sinput = Input(shape=(input_size,), name='state_input')
        ainput = Input(shape=(self.output_size,), name='action_input')
        s = Dense(hidden_layers[0], activation=activation_type)(sinput)
        a = Dense(hidden_layers[0], activation=activation_type)(ainput)
        x = concatenate([s, a])
        for idx in range(1, len(hidden_layers)):
            x = Dense(hidden_layers[idx], activation=activation_type)(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=[sinput, ainput], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

        return model

    def add_memory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def actor_optimizer(self):
        self.ainput = self.actor.input
        aoutput = self.actor.output
        trainable_weights = self.actor.trainable_weights
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, self.output_size))

        params_grad = tf.gradients(aoutput, trainable_weights, -self.action_gradient)
        grads = zip(params_grad, trainable_weights)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def critic_gradient(self):
        cinput = self.critic.input
        coutput = self.critic.output

        action_grads = k_b.gradients(coutput, cinput[1])
        return k_b.function([cinput[0], cinput[1]], action_grads)

    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def get_action(self, state_now, exploration_rate):

        exploration_rate = max(exploration_rate, 0.1)
        if random.random() > exploration_rate:
            action = self.actor.predict(state_now)[0]
            print('action' + str(action))
        else:
            action = np.array([])
            for idx in range(0, self.output_size):
                noise = random.random() * random.choice([-self.max_margin, self.max_margin])
                action = np.append(action, noise)

        return action

    def process_batch(self, batch_size):
        y = []
        states = []
        actions = []
        data = self.memory.getMiniBatch(batch_size)
        for idx in range(0, len(data)):
            states.append(data[idx]['state'])
            actions.append(data[idx]['action'])
            reward = data[idx]['reward']
            newState = data[idx]['newState']
            isFinal = data[idx]['isFinal']

            next_action = self.actor_target.predict(newState.reshape(1, self.input_size))

            q = self.critic_target.predict([newState.reshape(1, self.input_size), next_action])

            target = reward
            if not isFinal:
                target += self.discount_factor * q[0]
            y.append(target)

            if random.random() < 1:
                print ('@train@ \n state:{0} \n new_state{1} \n action:{2} \n reward{3} \n n_action{4} \n y{5}'.format(
                    states[idx], newState, actions[idx], reward, next_action, target))
        time.sleep(20)
        return np.array(states), np.array(actions), np.array(y)

    def update_target(self):
        update_rate = 0.25

        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        critic_target_weights = self.critic_target.get_weights()

        for i in range(len(actor_weights)):
            actor_target_weights[i] = update_rate * actor_weights[i] + (1 - update_rate) * actor_target_weights[i]

        for i in range(len(critic_weights)):
            critic_target_weights[i] = update_rate * critic_weights[i] + (1 - update_rate) * critic_target_weights[i]

    def learn_on_batch(self, batch_size):
        states, actions, y = self.process_batch(batch_size)
        # train critic NN
        self.critic.fit([states, actions], y, epochs=1, batch_size=batch_size, verbose=0)

        # train actor NN
        actor_next_actions = []
        for state in states:
            actor_next_action = self.actor.predict(state.reshape(1, self.input_size))[0]
            actor_next_actions.append(actor_next_action)
        actor_next_actions = np.array(actor_next_actions)

        # print(states)
        # print(actor_next_actions)
        a_grads = np.array(self.get_critic_grad([states, actor_next_actions]))[0]
        # print(a_grads)
        self.sess.run(self.opt, feed_dict={
            self.ainput: states,
            self.action_gradient: a_grads
        })


if __name__ == '__main__':
    inputSize = 12
    output = 3
    ddpg = DDPG(inputSize, output, 1000, 0.99, 0.001, 32, 2)
    ddpg.init_net_works([30, 30])

    step = 0
    for episode in range(100):
        observation = np.zeros(inputSize)
        for idx in range(inputSize):
            observation[idx] = random.random() * random.choice([-1, 1])
        for episode_step in range(200):
            observation = np.zeros(inputSize)
            for idx in range(inputSize):
                observation[idx] = random.random() * random.choice([-1, 1])
            action = ddpg.get_action(observation.reshape(1, inputSize), 0)
            n_observation = np.zeros(inputSize)
            for idx in range(inputSize):
                n_observation[idx] = random.random() * random.choice([-1, 1])
            n_action = ddpg.get_action(n_observation.reshape(1, inputSize), 0)
            reward = random.random()

            ddpg.add_memory(observation, action, reward, n_observation, True)
            step += 1
            if step > 32:
                ddpg.learn_on_batch(32)
        ddpg.update_target()
        print ('episode' + str(episode))
