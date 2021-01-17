from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.base_env import ActionTuple
import numpy as np
import datetime
import time
import math
from collections import deque
import os

import random
import torch
import torch.nn as nn
import CustomFuncionFor_mlAgent as CF

class VecModel(nn.Module):
    def __init__(self):
        super(VecModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(81,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,action_size),
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = out.detach().cpu().numpy()
        return out

    def predict(self, x):
        out = self.layer(x)
        out = out.argmax()
        out = out.detach().cpu().numpy()
        return out

class DQNAgent:
    def __init__(self, Factorlist):
        '''
        DQNAgentFactorlist.append(AgentsHelper)
        DQNAgentFactorlist.append(ConversionDataType)
        DQNAgentFactorlist.append(init_epsilon)
        DQNAgentFactorlist.append(min_epsilon)
        DQNAgentFactorlist.append(epsilon_decay)
        DQNAgentFactorlist.append(mem_maxlen)
        DQNAgentFactorlist.append(action_size)
        DQNAgentFactorlist.append(batch_size)
        DQNAgentFactorlist.append(discount_factor)
        DQNAgentFactorlist.append(lr)
        DQNAgemtFactorlist.append(load_model)
        DQNAgentFactorlist.append(load_path)
        '''
        self.AgentsHelper = Factorlist[0]
        self.ConversionDataType = Factorlist[1]
        self.epsilon = Factorlist[2]
        self.min_epsilon = Factorlist[3]
        self.epsilon_decay = Factorlist[4]
        self.memory = deque(maxlen=Factorlist[5])
        self.action_size = Factorlist[6]
        self.batch_size = Factorlist[7]
        self.discount_factor = Factorlist[8]
        self.lr = Factorlist[9]
        self.load_model = Factorlist[10]
        self.load_path = Factorlist[11]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VecModel = VecModel().to(self.device)
        self.targetVecModel = VecModel().to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.VecModel.parameters(), lr = self.lr)

        if self.load_model == True:
            print("model loaded!")
            self.VecModel.load_state_dict(torch.load(self.load_path))

    def get_action_from_visual_observation(self, vis_obs, behavior_name):
        #return action_tuple
        if self.epsilon > np.random.rand():
            actionList = []
            actionList.append(np.random.randint(0, self.action_size))
            return actionList

    def get_action_from_vector_observation(self, vec_obs, behavior_name):
        #return action_tuple
        if self.epsilon > np.random.rand():
            actionList = []
            actionList.append(np.random.randint(0, self.action_size))
            return actionList
        
        else:
            result = self.VecModel.predict(torch.tensor(vec_obs).to(self.device))
            actionList = []
            actionList.append(result)
            return actionList

    def append_sample(self, vec_obs, vis_obs, action, reward, n_vec_obs, n_vis_obs, done):
        #append_sample
        '''
        print(type(vec_obs), np.shape(vec_obs)) >>> <class 'numpy.ndarray'> (1, 81)
        print(type(vis_obs), np.shape(vis_obs)) >>> <class 'numpy.ndarray'> (1, 84, 84, 9)
        print(type(action), action) >>> <class 'list'> [2]
        print(type(reward)) >>> <class 'numpy.float32'>
        '''
        self.memory.append((vec_obs[0], vis_obs[0], action, reward, n_vec_obs[0], n_vis_obs[0], done))

    def updateTarget(self):
        self.targetVecModel.load_state_dict(self.VecModel.state_dict())
        print("Target network updated")

    def train_vis_model(self, done):
        if done:
            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay
        mini_batch = random.sample(self.memory, batch_size)

        vec_observations = []
        vis_observations = []
        actions = []
        rewards = []
        next_vec_observations = []
        next_vis_observations = []
        dones = []

        for i in range(batch_size):
            vec_observations.append(mini_batch[i][0])
            vis_observations.append(mini_batch[i][1])
            actions.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            next_vec_observations.append(mini_batch[i][4])
            next_vis_observations.append(mini_batch[i][5])
            dones.append(mini_batch[i][6])
        
        loss = 0
        return loss

    def train_vec_model(self, done):
        if done:
            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay
        mini_batch = random.sample(self.memory, batch_size)

        vec_observations = []
        vis_observations = []
        actions = []
        rewards = []
        next_vec_observations = []
        next_vis_observations = []
        dones = []

        for i in range(batch_size):
            vec_observations.append(mini_batch[i][0])
            vis_observations.append(mini_batch[i][1])
            actions.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            next_vec_observations.append(mini_batch[i][4])
            next_vis_observations.append(mini_batch[i][5])
            dones.append(mini_batch[i][6])
        

        vec_observations = np.array(vec_observations)
        next_vec_observations = np.array(next_vec_observations)
        target = self.VecModel.forward(torch.tensor(vec_observations).to(self.device))
        origintarget = target.copy()
        target_val = self.targetVecModel.forward(torch.tensor(next_vec_observations).to(self.device))

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        '''
        FallingStar.py:322: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; 
        use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, 
        `arr[np.array(seq)]`, which will result either in an error or a different result.
        target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])
        '''
        
        self.optimizer.zero_grad()
        loss = self.loss_func(torch.tensor(origintarget).to(self.device), torch.tensor(target).to(self.device))
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        return loss

if __name__ == "__main__":
    #Env_Setting
    string_log = CF.StringLogChannel()
    game = "FallingStar.exe"
    env_path = "./FallingStar/Build/"+game
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
    save_path = "./saved_model/"+game+"/"+date_time+"_DQN/model/"
    os.makedirs(save_path)
    load_path = "./saved_model/"+game+"/20210117-16-15-07_DQN/model/state_dict_model.pt"
    env = UnityEnvironment(file_name = env_path, side_channels = [string_log])
    env.reset()
    behavior_names = list(env.behavior_specs)
    ConversionDataType = CF.ConversionDataType()
    AgentsHelper = CF.AgentsHelper(env, string_log)
    AgentsHelper.print_specs_of_Agents(behavior_names)

    #Set Parameters...
    minEpisodeCount = 1
    trainEpisodeCount = 100
    testEpisodeCount = 100

    totalEpisodeCount = minEpisodeCount + trainEpisodeCount + testEpisodeCount
    trainEpisodeCount +=minEpisodeCount

    train_mode = True
    load_model = True

    init_epsilon = 1.0
    min_epsilon = 0.1
    lr = 0.00025
    action_size = 3
    mem_maxlen = 30000
    batch_size = 64
    discount_factor = 0.9
    epsilon_decay = 0.00005
    max_env_level = 2
    print_episode_interval = 3
    save_episode_interval = 3
    target_update_step = 10000

    NumOfAgent = len(behavior_names)
    vec_observations = []
    vis_observations = []
    next_vec_observations = []
    next_vis_observations = []
    actions = []
    rewards = []
    dones = []
    env_modes = []
    print(NumOfAgent)
    for i in range(NumOfAgent):
        vec_observations.append(0)
        vis_observations.append(0)
        next_vec_observations.append(0)
        next_vis_observations.append(0)
        actions.append(0)
        rewards.append(0)
        dones.append(False)
        env_modes.append(0)
    
    DQNAgentFactorlist = []
    DQNAgentFactorlist.append(AgentsHelper)
    DQNAgentFactorlist.append(ConversionDataType)
    DQNAgentFactorlist.append(init_epsilon)
    DQNAgentFactorlist.append(min_epsilon)
    DQNAgentFactorlist.append(epsilon_decay)
    DQNAgentFactorlist.append(mem_maxlen)
    DQNAgentFactorlist.append(action_size)
    DQNAgentFactorlist.append(batch_size)
    DQNAgentFactorlist.append(discount_factor)
    DQNAgentFactorlist.append(lr)
    DQNAgentFactorlist.append(load_model)
    DQNAgentFactorlist.append(load_path)
    
    DQNAgent = DQNAgent(DQNAgentFactorlist)
    AgentsHelper.SendMessageToEnv("Env Connection Test")

    #run episodes!
    #에피소드 카운트가 늘어나는 기준은 0번 에이전트의 에피소드가 종료 될 때마다이다.
    totalStep = 0
    
    episodelosses = []
    episodeRewards = []
    for i in range(NumOfAgent):
        episodeRewards.append(0)

    #tracemalloc.start(10)
    for episodeCount in range(totalEpisodeCount):
        #time1 = tracemalloc.take_snapshot()
        print("episodeCount+")
        dones[0] = False
        if episodeCount > trainEpisodeCount:
            train_mode = False
        
        while not dones[0]:
            totalStep +=1
            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
                vec_observation, vis_observation, done = AgentsHelper.getObservation_lite(behavior_name)
                vec_observations[behavior_name_Num] = vec_observation
                vis_observations[behavior_name_Num] = vis_observation
                action = DQNAgent.get_action_from_vector_observation(vec_observation, behavior_name)
                actionTuple = ConversionDataType.ConvertList2DiscreteAction(action,behavior_name)
                env.set_actions(behavior_name, actionTuple)
                actions[behavior_name_Num] = action

            env.step()

            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
                next_vec_observation, next_vis_observation, done = AgentsHelper.getObservation_lite(behavior_name)
                next_vec_observations[behavior_name_Num] = next_vec_observation
                next_vis_observations[behavior_name_Num] = next_vis_observation
                reward = AgentsHelper.get_reward(behavior_name)
                rewards[behavior_name_Num] = reward
                dones[behavior_name_Num] = done

            for behavior_name in behavior_names:
                behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
                if train_mode == True:
                    DQNAgent.append_sample(
                        vec_observations[behavior_name_Num],
                        vis_observations[behavior_name_Num],
                        actions[behavior_name_Num],
                        rewards[behavior_name_Num],
                        next_vec_observations[behavior_name_Num],
                        next_vis_observations[behavior_name_Num],
                        dones[behavior_name_Num],
                    )
                else:
                    time.sleep(0.01)
                    DQNAgent.epsilon = 0.05

                episodeRewards[behavior_name_Num] += rewards[behavior_name_Num]

            if episodeCount>minEpisodeCount and train_mode == True:
                loss = DQNAgent.train_vec_model(dones[0])
                episodelosses.append(loss)

                if totalStep % (target_update_step) == 0:
                    DQNAgent.updateTarget()
        
        if episodeCount % save_episode_interval == 0 and episodeCount !=0:
            torch.save(DQNAgent.VecModel.state_dict(), save_path+"state_dict_model.pt")

        if episodeCount % print_episode_interval == 0 and episodeCount !=0:
            print("step:{}//average loss = {:.3f}, average rewards = {:.2f}|{:.2f}|{:.2f}, epsilon = {:.4f}".format(
                totalStep,
                np.mean(episodelosses), 
                np.mean(episodeRewards[0]), 
                np.mean(episodeRewards[1]), 
                np.mean(episodeRewards[2]), 
                DQNAgent.epsilon))
            episodelosses = []
            episodeRewards = []
            for i in range(NumOfAgent):
                episodeRewards.append(0)
        #time2 = tracemalloc.take_snapshot()
        #stats = time2.compare_to(time1,'lineno')
        #for stat in stats[:3]:
            #print(stat)



            
        



    '''
    for i in range(200):
        for behavior_name in behavior_names:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
            vec_observation, vis_observation = AgentsHelper.getObservation(behavior_name)
            vec_observations[behavior_name_Num] = vec_observation
            vis_observations[behavior_name_Num] = vis_observation
        #print("vec_observations_shape: ", np.shape(vec_observations))#  >>>(3,1,81)
        #print("vis_observations_shape ", np.shape(vis_observations))#  >>>(3,1,84,84,9)

        sendMessage = ""
        for index, env_mode in enumerate(env_modes):
            sendMessage +=str(index) + "?" + str(env_mode)+"/"
        sendMessage = ConversionDataType.delete_last_char(sendMessage)
        string_log.send_string("@" + sendMessage)
        env.step()

    '''
    env.close()