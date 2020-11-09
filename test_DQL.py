#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 08:14:40 2020

@author: Phuonglun
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import Counter, deque

tf.compat.v1.reset_default_graph()
#build Q network with 3 convolutional layers

def Q_network(X, name_scope):
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0)
    with tf.variable_scope(name_scope) as scope: 
        #initialize convolutional layer
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8,8), stride=4, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_1',layer_1)
    
        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4,4), stride=2, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_2',layer_2)
    
        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3,3), stride=1, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_3', layer_3)
    
        #flatten the result of layer 3 before feeding to the fully connected layer
        flat = flatten(layer_3)
    
        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('fc', fc)
    
        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output', output)
    
    #Var will store the network parameters such as weights
    
        vars = {v.name[len(scope.name):]:v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        return vars, output

epsilon = 0.5
eps_min = 0.05
eps_max = 1
eps_decay_steps = 50000

def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
    if np.random.rand()<epsilon:
        return np.random.randint(n_outputs)
    else:
        return action
    
buffer_len = 2000
exp_buffer = deque(maxlen = buffer_len)

#sample the experiences from memory
#batch size is the number of experience sampled from memory
def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]

#define nerwork hyperparameters
num_episode = 1
batch_size = 20
input_shape = (None, 88,80,1)
learning_rate = 0.001
X_shape = (None, 88,80,1)
discount_factor = 0.97

global_step = 0
copy_steps= 100
steps_train = 4
start_steps = 2000
n_outputs = 10

logdir = 'logs'
tf.compat.v1.reset_default_graph()

X=tf.compat.v1.placeholder(tf.float32, shape=X_shape)

#build our Q-network, which take input X  and generate Q values for all the actions
mainQ, mainQ_outputs = Q_network(X,'mainQ')

#similarly build target Q Network
targetQ, targetQ_outputs = Q_network(X, 'targetQ')

#define placeholder for our action values
X_action = tf.compat.v1.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs*tf.one_hot(X_action,n_outputs), axis=-1, keep_dims=True)

#copy primary Q paramters to target Q network parameters
copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

#define placeholder for our output i.e., action
y = tf.compat.v1.placeholder(tf.float32, shape=(None,1))

#calculate the loss which is difference between actual value and predicted value
loss = tf.reduce_mean(tf.square(y-Q_action))

#we use adam optimization for minimizing the loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()

num_uavs = 2
area = 10 #meters in length and width
num_ues = 5

#generate ues positions
coord_ues =[[np.random.random()*area, np.random.random()*area]]
for i in range(num_ues-1):
    generate_coord = [np.random.random()*area, np.random.random()*area]
    coord_ues.append(generate_coord)

#now we start the tensor session and run the model
with tf.Session() as sess:
    init.run()
    
    #for each episode
    for i in range(num_episode):
        done = False
        #generate the initial UAVs positions
        obs = [[np.random.random()*area, np.random.random()*area]]
        for i in range(num_uavs-1):
            generate_coord = [np.random.random()*area, np.random.random()*area]
            obs.append(generate_coord)
        epoch = 0
        episodic_reward= 0
        actions_counter = Counter()
        episodic_loss = []
        
        while not done:
            #feed state and get Q values for each actions
            actions = mainQ_outputs.eval(feed_dict={X:obs, in_training_mode:False})
             #get action
            action = np.argmax(actions,axis=-1)
            actions_counter[str(action)] += 1
            
            #select the action using epsilon greedy policy
            action = epsilon_greedy(action, global_step)
            
            #now perform the action and get to the next state, next_obs, received reward
            for i in range(len(obs)):
                next_obs_each_uav = [sum(x) for x in zip(obs[i],action[i])]
                if i==0:
                    next_obs = [next_obs_each_uav]
                else:
                    next_obs.append(next_obs_each_uav)
            
            reward = 1
            
            #store this transition as an experience in the replay buffer
            exp_buffer.append([obs, action, next_obs, reward, done])
            
            #after certain steps, we train our Q network with samples from the experience replay buffer
            if global_step % steps_train == 0 and global_step > start_steps:
                #sample experience
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
                
                #next action
                next_act = mainQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})
                
                #reward
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1-o_done)
                
                #merge all summaries
                mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch,axis=-1), X_action:o_act, in_training_mode:False})
                
                #now we train the network and calculate the loss
                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
           
            #after some interval we copy our main Q network weights to target Q network
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()
            
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
            print('epoch', epoch, 'reward', episodic_reward,)
            
                
                
            
        


 
    

    
    
    