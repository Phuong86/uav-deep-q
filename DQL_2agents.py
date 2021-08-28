# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 22:03:12 2019

@author: pluong2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:40:53 2019

@author: pluong2
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from collections import deque, Counter
import random
from datetime import datetime
import math
import matplotlib.pyplot as plt

x_max =300
y_max =300
H=150
R=200
n_uavs = 2
n_ues = 12
#generate 4ues in area (10mx10m)
ue_pos = [random.randint(0, x_max) for i in range(2*n_ues)]
#ue_pos=[4, 1, 0, 1, 2, 2, 4, 4]
#ue_pos = [3,3,2,1,4,4,3,2]
Com_range = np.sqrt(H**2+R**2)
step_resolution = 10
actions_1uav = [(0,step_resolution),(step_resolution,0),(0,-step_resolution),(-step_resolution,0),(0,0)]
actions_space =[]
for i in range(len(actions_1uav)):
    for j in range(len(actions_1uav)):
        actions_space.append(list(actions_1uav[i]+actions_1uav[j]))
n_outputs = len(actions_space)


tf.reset_default_graph()

def q_network(X, name_scope):
    #initialize layers
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    with tf.variable_scope(name_scope) as scope:
        #feed input state to fully connected layer
        layer_1 = fully_connected(X, num_outputs=32, weights_initializer=initializer)
        tf.summary.histogram('layer_1',layer_1)
        
        layer_2 = fully_connected(layer_1, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('layer_2',layer_2)
        
#        layer_3 = fully_connected(layer_2, num_outputs=64, weights_initializer=initializer)
#        tf.summary.histogram('layer_2',layer_2)
#        
#        layer_4 = fully_connected(layer_3, num_outputs=64, weights_initializer=initializer)
#        tf.summary.histogram('layer_2',layer_2)
        
        output = fully_connected(layer_2, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output',output)
        
        #Vars will store the parameters of the network such as weights 
        
        vars = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
        return vars, output


epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_step = 500000

def epsilon_greedy(action,step):
#    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_step)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action[0]

#define replay buffer with length 20000 to hold the experience 
buffer_len = 2000
exp_buffer = deque(maxlen=buffer_len)


#define function to sample the experiences from the memory
def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3]

def distance(ue_position,state_uavs,height):
    dis = []
    j=0
    for i in range(int(len(state_uavs)/2)):
        uav_pos = []
        uav_pos.append(state_uavs[j])
        uav_pos.append(state_uavs[j+1])
        j+=2
        dis.append(math.sqrt((uav_pos[0]-ue_position[0])**2+(uav_pos[1]-ue_position[1])**2+height**2)) 
    return dis


def func_rewards_for_user(max_dis,dis_uav_ue):
    dis_ = [i for i in dis_uav_ue if i<=max_dis]
    if len(dis_)==0:
        r=-2
    elif len(dis_)==1:
        r=1
    else:
        r=1+len(dis_)*1.5
    return r
        

def func_rewards_num_users(max_dis,dis_uav_ue):
    dis_ = [i for i in dis_uav_ue if i<=max_dis]
    if len(dis_)==0:
        r=0
    elif len(dis_)==1:
        r=1
    else:
        r=1+len(dis_)*1.5
    return r


#define network hyperparameters
num_episodes = 800
batch_size = 48
input_shape = (1,4)
learning_rate = 0.01
X_shape = (None,4)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
max_steps = 1000

#tf.reset_default_graph()
#define placeholder for input
X = tf.placeholder(tf.float32, shape = X_shape)

in_training_mode = tf.placeholder(tf.bool)

mainQ, mainQ_outputs = q_network(X, 'mainQ')
targetQ, targetQ_outputs = q_network(X, 'targetQ')

X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)

"""Copy the primary Q network parameters to the target Q network"""
copy_op = [tf.assign(main_name,targetQ[var_name]) for var_name, main_name in mainQ.items()]

copy_target_to_main = tf.group(*copy_op)

y=tf.placeholder(tf.float32, shape = (None,1))

""" now we calculate the loss which is the difference between actual value and predicted value"""
loss = tf.reduce_mean(tf.square(y-Q_action))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
#sess = tf.InteractiveSession()
#with sess.as_default():
    init.run()
    timestep_reward = []
    #for each episode
    episodic_loss = [0]
    fini_loss = []
    
    for i in range(num_episodes):
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        #episodic_loss = []
        
        obs = [random.randint(0, x_max) for i in range(2*n_uavs)]
        obs = np.array(obs).reshape((4,))
        reward_store=[]
        
        
        t = 0
        
        total_reward = 0
        while t < max_steps:
            t +=1
        
            actions = mainQ_outputs.eval(feed_dict={X:[obs], in_training_mode:False})
            
            #get action
            action = np.argmax(actions,axis=-1)
#            print(f"argmax,{action}")
            actions_counter[str(action)]+=1
            
            action = epsilon_greedy(action,global_step)
            
            reward_temp = 0
            n_iter=0
            for i in range(int(len(ue_pos)/2)):
                
                one_ue_pos = []
                one_ue_pos.append(ue_pos[n_iter])
                one_ue_pos.append(ue_pos[n_iter+1])
                n_iter+=2
                reward_temp += func_rewards_for_user(Com_range,distance(one_ue_pos,obs,H))
            reward_store.append(reward_temp)
            
#            print(f"greedy,{action}")
            """now perform the action and move to the next state, next_obs, receive reward"""
#            print(f"here,{action}")
            next_obs = [sum(i) for i in zip(obs,actions_space[action])]
            """calculate the reward"""
            reward = 0
            real_reward = 0
            j=0
            for i in range(int(len(ue_pos)/2)):
                
                one_ue_pos = []
                one_ue_pos.append(ue_pos[j])
                one_ue_pos.append(ue_pos[j+1])
                j+=2
                real_reward += func_rewards_for_user(Com_range,distance(one_ue_pos,next_obs,H))
            reward_store.append(real_reward)
            if reward_store[len(reward_store)-1]<reward_store[len(reward_store)-2]:
                reward += -10
            elif reward_store[len(reward_store)-1]>reward_store[len(reward_store)-2]:
                reward += 10
            else:
                reward = 0 
            #check if next_states contains terminal states 
            check_obs = [i for i in next_obs if i>300 or i<0]
            if len(check_obs)>0:
                """penalty the reward"""
                reward+=-50
#                """cancel movement of uav and update new observation"""
                next_obs = obs 
            
            total_reward += reward
            """store this transition as an experience in the replay buffer"""
            
            exp_buffer.append([obs,action,next_obs,reward])
            
            if global_step % steps_train ==0 and global_step > start_steps:
                o_obs,o_act,o_next_obs,o_reward = sample_memories(batch_size)
#                for x in range(len(o_obs)):
#                    o_obs = o_obs[x]
#                    o_next_obs = o_next_obs[x]
                o_obs = [x for x in o_obs]
                o_next_obs = [x for x in o_next_obs]
#                for x in range(len(o_obs)):
                next_act = mainQ_outputs.eval(feed_dict={X:o_next_obs, in_training_mode:False})
                
                """reward"""
                y_batch = o_reward + discount_factor*np.max(next_act,axis=-1)
                
                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
            if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()
#                
            obs=next_obs
#            #print(f"next obs,{obs}")
            epoch += 1
            global_step += 1
            episodic_reward += reward
        timestep_reward.append(total_reward)
        fini_loss.append(episodic_loss[len(episodic_loss)-1])
        saver.save(sess, "./2UAVs_positions")
        print('Epoch', epoch, 'Reward', episodic_reward,)

plt.plot(timestep_reward)        
#        """testing"""
#     num_epoch = 100
#     global_step_test = 0
#     timestep_reward_test = []
#     actions_counter_test = Counter()
#     obs_test = [random.randint(0, x_max) for i in range(2*n_uavs)]
#     obs_test = np.array(obs_test).reshape((4,))
#     total_reward_test = 0
#     epoch_test = 0
#     reward_test_epoch =[]
#     reward_store_test=[0]
#     for i in range(num_epoch):
    
    
    
#            saver.restore(sess, "./2UAVs_positions")          actions_test = mainQ_outputs.eval(feed_dict={X:[obs_test], in_training_mode:False})
            
#             #get action
#         action_test = np.argmax(actions_test,axis=-1)
#         action_test=action_test[0]
#         #            print(f"argmax,{action}")
#         actions_counter_test[str(action_test)]+=1
            
#         #action_test = epsilon_greedy(action_test,global_step_test)    
        
#             #while the state is not terminal state
#         next_obs_test = [sum(i) for i in zip(obs_test,actions_space[action_test])]
#         """calculate the reward"""
#         real_reward_test = 0
#         reward_test = 0
#         j=0
#         for i in range(int(len(ue_pos)/2)):
                
#             one_ue_pos = []
#             one_ue_pos.append(ue_pos[j])
#             one_ue_pos.append(ue_pos[j+1])
#             j+=2
#             real_reward_test += func_rewards_num_users(Com_range,distance(one_ue_pos,next_obs_test,H))
#         reward_store_test.append(real_reward_test)
# #        if reward_store_test[len(reward_store_test)-1]<reward_store_test[len(reward_store_test)-2]:
# #            reward_test +=  0
# #        elif reward_store_test[len(reward_store_test)-1]>reward_store_test[len(reward_store_test)-2]:
# #            reward_test +=  real_reward_test
# #        else:
# #            reward_test = real_reward_test
#         #check if next_states contains terminal states 
#         check_obs_test = [i for i in next_obs_test if i>300 or i<0]
#         if len(check_obs_test)>0:
#             """penalty the reward"""
#             #reward_test += -50
#         #                """cancel movement of uav and update new observation"""
#             next_obs_test = obs_test
                
#         total_reward_test += real_reward_test 
#         obs_test=next_obs_test
#         global_step_test +=1
#         epoch_test += 1
#         reward_test_epoch.append(real_reward_test)
#         timestep_reward_test.append(total_reward_test)
#         print('Epoch test', epoch_test, 'Reward test', real_reward_test,) 
# plt.plot(real_reward_test)
#plt.plot(episodic_loss)
            
       
        







