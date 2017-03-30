import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import mxnet as mx
import scipy.signal
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont
#%matplotlib inline
from helper import *
import argparse
from random import choice
from time import sleep
from time import time

parser = argparse.ArgumentParser(description='Learning to reinforcement learn')
parser.add_argument('--num-hidden', type=int, default=48, help='the number of hidden nodes')
parser.add_argument('--num-layers', type=int, default=1, help='the number of layers')
parser.add_argument('--num-threads', type=int, default=7, help='the number of threads')
parser.add_argument('--episode-len', type=int, default=100, help='the number of threads')

args = parser.parse_args()
class dependent_bandit():
    def __init__(self,difficulty):
        self.num_actions = 2
        self.difficulty = difficulty
        self.reset()
        
    def set_restless_prob(self):
        self.bandit = np.array([self.restless_list[self.timestep],1 - self.restless_list[self.timestep]])
        
    def reset(self):
        self.timestep = 0
        if self.difficulty == 'restless': 
            variance = np.random.uniform(0,.5)
            self.restless_list = np.cumsum(np.random.uniform(-variance,variance,(150,1)))
            self.restless_list = (self.restless_list - np.min(self.restless_list)) / (np.max(self.restless_list - np.min(self.restless_list))) 
            self.set_restless_prob()
        if self.difficulty == 'easy': bandit_prob = np.random.choice([0.9,0.1])
        if self.difficulty == 'medium': bandit_prob = np.random.choice([0.75,0.25])
        if self.difficulty == 'hard': bandit_prob = np.random.choice([0.6,0.4])
        if self.difficulty == 'uniform': bandit_prob = np.random.uniform()
        if self.difficulty != 'independent' and self.difficulty != 'restless':
            self.bandit = np.array([bandit_prob,1 - bandit_prob])
        else:
            self.bandit = np.random.uniform(size=2)
        
    def pullArm(self,action):
        #Get a random number.
        if self.difficulty == 'restless': self.set_restless_prob()
        self.timestep += 1
        bandit = self.bandit[action]
        result = np.random.uniform()
        if result < bandit:
            #return a positive reward.
            reward = 1
        else:
            #return a negative reward.
            reward = 0
        if self.timestep > args.episode_len-1: 
            done = True
        else: done = False
        return reward,done,self.timestep
        
        
def AC_Network(a_size,scope):
    prev_rewards = mx.symbol.Cast(data=mx.symbol.Variable('prev_rewards'), dtype='float32')
    prev_actions = mx.symbol.Cast(data=mx.symbol.Variable('prev_actions'), dtype='int32')
    timestep = mx.symbol.Cast(data=mx.symbol.Variable('timestep'), dtype='float32')
    prev_actions_onehot=mx.symbol.one_hot(indices=prev_actions,depth=a_size, dtype='float32')
    
    hidden=mx.symbol.Concat(prev_rewards,prev_actions_onehot,timestep,dim=1)
    rnn_in = mx.symbol.expand_dims(hidden, axis=0) 
    
    stack = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))                    
    
    if scope == 'train':
        seq_len =  args.episode_len      
        outputs, states = stack.unroll(seq_len, inputs=rnn_in, merge_outputs=True)        
        rnn_out=mx.symbol.Reshape(outputs, shape=(-1, args.num_hidden))        
        fc_policy=mx.symbol.FullyConnected(data=rnn_out,num_hidden=a_size,name='fc_policy')
        policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
        value = mx.symbol.FullyConnected(data=rnn_out, name='fc_value', num_hidden=1)
        
        actions =mx.symbol.Cast(data=mx.symbol.Variable('actions'), dtype='int32')
        actions_onehot=mx.symbol.one_hot(indices=actions,depth=a_size)
        target_v = mx.symbol.Cast(data=mx.symbol.Variable('target_v'), dtype='float32')
        advantages = mx.symbol.Cast(data=mx.symbol.Variable('advantages'), dtype='float32')
        responsible_outputs = mx.symbol.sum(data=policy * actions_onehot, axis=1)
        #Loss functions
        value_loss = 0.5 * mx.symbol.sum(mx.symbol.square(target_v - mx.symbol.Reshape(value,shape=(-1,))))
        entropy = - mx.symbol.sum(data=policy * mx.symbol.log(policy + 1e-7))
        policy_loss = -mx.symbol.sum(mx.symbol.log(responsible_outputs + 1e-7)*advantages)
        output = 0.5 *value_loss + policy_loss + entropy * 0.05
        loss=mx.sym.MakeLoss(output)
        return loss
    if scope == 'episode':
        c_in = mx.symbol.Cast(data=mx.symbol.Variable('c_in'), dtype='float32')
        h_in = mx.symbol.Cast(data=mx.symbol.Variable('h_in'), dtype='float32')
        state_in = [c_in, h_in]
        seq_len = 1   
        outputs, states = stack.unroll(seq_len, inputs=rnn_in,begin_state=state_in, merge_outputs=True)
        rnn_out=mx.symbol.Reshape(outputs, shape=(-1, args.num_hidden))        
        fc_policy=mx.symbol.FullyConnected(data=rnn_out,num_hidden=a_size,name='fc_policy')
        policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
        value = mx.symbol.FullyConnected(data=rnn_out, name='fc_value', num_hidden=1)
        return mx.symbol.Group([policy, value, states[0],states[1]])
    if scope == 'test':
        c_in = mx.symbol.Cast(data=mx.symbol.Variable('c_in'), dtype='float32')
        h_in = mx.symbol.Cast(data=mx.symbol.Variable('h_in'), dtype='float32')
        state_in = [c_in, h_in]
        seq_len = 1  
        outputs, states = stack.unroll(seq_len, inputs=rnn_in,begin_state=state_in, merge_outputs=True)
        rnn_out=mx.symbol.Reshape(outputs, shape=(-1, args.num_hidden))        
        fc_policy=mx.symbol.FullyConnected(data=rnn_out,num_hidden=a_size,name='fc_policy')
        policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
        return policy
        
    
            
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
    
def createNetwork(a_size, ctx, phase='train', bef_args=None):
    if phase=='train':
        modtr = mx.mod.Module(symbol=AC_Network(a_size,'train'), data_names=('prev_rewards','prev_actions','timestep',
        'actions','target_v','advantages'), label_names=None, context=ctx)
        batch=args.episode_len
        modtr.bind(data_shapes=[('prev_rewards',(batch,1)),('prev_actions',(batch,)),('timestep',(batch,1)),
        ('actions',(batch,)),('target_v',(batch,)),('advantages',(batch,))],label_shapes=None,for_training=True)
        modtr.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)
        modtr.init_optimizer(
            optimizer='adam',
            optimizer_params={
                'learning_rate': 0.0002,
                'wd': 0.,
                'beta1': 0.5,
        })
        return modtr
    elif phase=='episode':
        modep = mx.mod.Module(symbol=AC_Network(a_size,'episode'), data_names=('prev_rewards','prev_actions','timestep','c_in','h_in'), 
        label_names=None, context=ctx)
        batch=1
        modep.bind(data_shapes=[('prev_rewards',(batch,1)),('prev_actions',(batch,)),('timestep',(batch,1)),('c_in',(1,args.num_hidden)),
        ('h_in',(1,args.num_hidden))],label_shapes=None,for_training=False)
        modep.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)
        return modep        
    else:
        modte = mx.mod.Module(symbol=AC_Network(a_size,'test'), data_names=('prev_rewards','prev_actions','timestep','c_in','h_in'), 
        label_names=None, context=ctx)
        batch=1
        modte.bind(data_shapes=[('prev_rewards',(batch,1)),('prev_actions',(batch,)),('timestep',(batch,1)),('c_in',(1,args.num_hidden)),
        ('h_in',(1,args.num_hidden))],label_shapes=None,for_training=False)

        modte.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),arg_params=bef_args)
        return modte
 
    
class Worker():
    def __init__(self,game,name,a_size,model_path):
        self.name = "worker_" + str(name)
        self.number = name              
        self.model_path = model_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.ctx = mx.cpu(name)
        self.env = game
        self.a_size=a_size
        
        
    def train(self,rollout,gamma,bootstrap_value):
        global master_network,lock                                
        rollout = np.array(rollout)
        actions = rollout[:,0]
        rewards = rollout[:,1]
        timesteps = rollout[:,2]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:,4]
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        lock.acquire()
        loss=master_network.forward(mx.io.DataBatch([mx.nd.array(np.vstack(prev_rewards),self.ctx), mx.nd.array(prev_actions,self.ctx), 
            mx.nd.array(np.vstack(timesteps),self.ctx),mx.nd.array(actions,self.ctx),
            mx.nd.array(discounted_rewards,self.ctx),mx.nd.array(advantages,self.ctx)],[]),is_train=True)
        master_network.backward()
        master_network.update()
        arg_params,aux_params=master_network.get_params()
        episode_net.set_params(arg_params=arg_params,aux_params=aux_params) 
        lock.release()
        return loss
        
    def work(self,gamma,save_prefix,istrain):
        global episode_count,Tmax,lock,master_network,episode_net
        total_steps = 0
        lock.acquire()
        print "Starting worker " + str(self.number)
        lock.release()
               
        while episode_count < Tmax:            
            episode_buffer = []
            episode_values = []
            episode_frames = []
            episode_reward = [0,0]
            episode_step_count = 0
            d = False
            r = 0
            a = 0
            t = 0
            self.env.reset()            
            
            c_init = mx.nd.array(np.zeros((1, args.num_hidden), np.float32),self.ctx)
            h_init = mx.nd.array(np.zeros((1, args.num_hidden), np.float32),self.ctx)
            state_init = [c_init, h_init]
            rnn_state = state_init            
            
            lock.acquire()           
            while d == False:
                #Take an action using probabilities from policy network output.
                #Batch = namedtuple('Batch', ['prev_rewards','prev_actions','timestep','c_in','h_in'])
                
                episode_net.forward(mx.io.DataBatch([mx.nd.array([[r]],self.ctx), mx.nd.array([a],self.ctx),
                    mx.nd.array([[t]],self.ctx),rnn_state[0], rnn_state[1]],[]), is_train=False)
                a_dist,v,rnn_state_new1,rnn_state_new2=episode_net.get_outputs()
                
                #a = np.random.choice(a_dist.asnumpy().reshape((-1)),p=a_dist.asnumpy().reshape((-1)))
                #a = np.argmax(a_dist == a)                
                a = np.random.choice(2,p=a_dist.asnumpy().reshape((-1)))
                rnn_state = [rnn_state_new1,rnn_state_new2]
                
                r,d,t = self.env.pullArm(a)                                
                episode_buffer.append([a,r,t,d,v.asnumpy()[0][0]])
                episode_values.append(v.asnumpy()[0][0])
                episode_frames.append(set_image_bandit(episode_reward,self.env.bandit,a,t))
                episode_reward[a] += r
                total_steps += 1
                episode_step_count += 1
            lock.release()    
                                        
            self.episode_rewards.append(np.sum(episode_reward))
            self.episode_lengths.append(episode_step_count)
            self.episode_mean_values.append(np.mean(episode_values))
            
            # Update the network using the experience buffer at the end of the episode.
            if len(episode_buffer) != 0 and istrain == True:
                loss = self.train(episode_buffer,gamma,0.0)
        
                
            # Periodically save gifs of episodes, model parameters, and summary statistics.
            lock.acquire()
            if episode_count % 50 == 0 and episode_count != 0:
                if episode_count % 5000 == 0 and istrain == True:
                    master_network.save_params('%s-%04d.params'%(save_prefix, episode_count))
                    
                if episode_count % 5000 == 0 :
                    print len(episode_frames)
                    self.images = np.array(episode_frames)
                    make_gif(self.images,'./frames/image'+str(episode_count)+'.gif',
                        duration=len(self.images)*0.1,true_image=True)
                mean_reward = np.mean(self.episode_rewards[-50:])
                mean_length = np.mean(self.episode_lengths[-50:])
                mean_value = np.mean(self.episode_mean_values[-50:])
                print "mean reward: %f mean length: %f mean value: %f" %(mean_reward, mean_length, mean_value)
            
            episode_count += 1
            lock.release()            
            
                
gamma = .8 # discount rate for advantage estimation and reward discounting
a_size = 2 
load_model = True
istrain = True
model_path = './'
save_prefix= "a3c-bandit"

master_network = createNetwork(a_size,phase='train',ctx=mx.cpu(0))
episode_net = createNetwork(a_size,ctx=mx.cpu(0),phase='episode') 
Tmax=1000000
lock = threading.Lock()
num_workers=args.num_threads
workers = []
episode_count=0
for i in range(1,num_workers):
        workers.append(Worker(dependent_bandit('medium'),i,a_size,model_path))
threads = []
for worker in workers:
    worker_work = lambda: worker.work(gamma,save_prefix,istrain)
    thread=threading.Thread(target=(worker_work))
    thread.start()
    for i in range(100000):
        i
    threads.append(thread)
for t in threads:
    t.join()
