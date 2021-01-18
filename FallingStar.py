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

#Env_Setting
string_log = CF.StringLogChannel()
game = "FallingStar.exe"
env_path = "./FallingStar/Build/"+game
date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_model/"+game+"/"+date_time+"_DQN/model/"
os.makedirs(save_path)
load_path = "./saved_model/"+game+"/20210117-16-15-07_DQN/model/"
env = UnityEnvironment(file_name = env_path, side_channels = [string_log])
env.reset()
behavior_names = list(env.behavior_specs)
ConversionDataType = CF.ConversionDataType()
AgentsHelper = CF.AgentsHelper(env, string_log, ConversionDataType)
AgentsHelper.print_specs_of_Agents(behavior_names)

#Set Parameters...
minEpisodeCount = 20
trainEpisodeCount = 1000
testEpisodeCount = 1000
levelUpEpisodeCount = 200

totalEpisodeCount = minEpisodeCount + trainEpisodeCount + testEpisodeCount
trainEpisodeCount +=minEpisodeCount

train_mode = True
load_model = False

init_epsilon = 1.0
min_epsilon = 0.1
lr = 0.00025
action_size = 3
mem_maxlen = 30000
batch_size = 64
discount_factor = 0.9
epsilon_decay = 0.00005
max_env_level = 2
print_episode_interval = 50
save_episode_interval = 100
target_update_step = 10000

actionWith_visModelPredict = False


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
    
    def forward(self, x, device):
        x = torch.tensor(x).to(device)
        out = self.layer(x)
        out = out.detach().cpu().numpy()
        return out

    def predict(self, x, device):
        x = torch.tensor(x).to(device)
        out = self.layer(x)
        out = out.argmax()
        out = out.detach().cpu().numpy()
        return out

class VisModel(nn.Module):
    def __init__(self):
        super(VisModel, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(9,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.layer = nn.Sequential(
            nn.Linear(18496,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,action_size),
        )
    
    def forward(self, x, device):
        x = (np.array(x, dtype='float32')-(255.0/2))/(255.0/2)
        x = torch.tensor(x).to(device)
        out = self.convlayer(x)
        out = out.view(batch_size,-1)
        out = self.layer(out)
        out = out.detach().cpu().numpy()
        return out

    def predict(self, x, device):
        x = (np.array(x, dtype='float32')-(255.0/2))/(255.0/2)
        x = torch.tensor(x).to(device)
        out = self.convlayer(x)
        out = out.view(batch_size,-1)
        out = self.layer(out)
        out = out.argmax()
        out = out.detach().cpu().numpy()
        return out

class DQNAgent:
    def __init__(self):
        self.AgentsHelper = AgentsHelper
        self.ConversionDataType = ConversionDataType
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=mem_maxlen)
        self.action_size = action_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.lr = lr
        self.load_model = load_model
        self.load_path = load_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VecModel = VecModel().to(self.device)
        self.VisModel = VisModel().to(self.device)
        self.targetVecModel = VecModel().to(self.device)
        self.targetVisModel = VisModel().to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.VecModel.parameters(), lr = self.lr)
        self.vis_optimizer = torch.optim.Adam(self.VisModel.parameters(), lr = self.lr)

        if self.load_model == True:
            print("model loaded!")
            self.VecModel.load_state_dict(torch.load(self.load_path+"state_dict_model.pt"))
            self.VisModel.load_state_dict(torch.load(self.load_path+"vis_state_dict_model.pt"))

    def get_action_from_visual_observation(self, vis_obs, behavior_name):
        #return action_tuple
        if self.epsilon > np.random.rand():
            actionList = []
            actionList.append(np.random.randint(0, self.action_size))
            return actionList
        else:
            result = self.VisModel.predict(vis_obs, self.device)
            actionList = []
            actionList.append(result)
            return actionList

    def get_action_from_vector_observation(self, vec_obs, behavior_name):
        #return action_tuple
        if self.epsilon > np.random.rand():
            actionList = []
            actionList.append(np.random.randint(0, self.action_size))
            return actionList
        else:
            result = self.VecModel.predict(vec_obs, self.device)
            actionList = []
            actionList.append(result)
            return actionList

    def append_sample(self, vec_obs, vis_obs, action, reward, n_vec_obs, n_vis_obs, done):
        self.memory.append((vec_obs[0], vis_obs, action, reward, n_vec_obs[0], n_vis_obs, done))

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

        vis_observations = np.array(vis_observations)
        next_vis_observations = np.array(next_vis_observations)
        target = self.VisModel.forward(vis_observations, self.device)
        origintarget = target.copy()
        target_val = self.targetVisModel.forward(vis_observations, self.device)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])
        
        self.vis_optimizer.zero_grad()
        loss = self.loss_func(torch.tensor(origintarget).to(self.device), torch.tensor(target).to(self.device))
        loss.requires_grad = True
        loss.backward()
        self.vis_optimizer.step()
        
        loss = loss.detach().cpu().numpy()
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
        target = self.VecModel.forward(vec_observations, self.device)
        origintarget = target.copy()
        target_val = self.targetVecModel.forward(vec_observations, self.device)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])
        
        self.optimizer.zero_grad()
        loss = self.loss_func(torch.tensor(origintarget).to(self.device), torch.tensor(target).to(self.device))
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
        
        loss = loss.detach().cpu().numpy()
        return loss

if __name__ == "__main__":
    NumOfAgent = len(behavior_names)
    vec_observations = []
    vis_observations = []
    next_vec_observations = []
    next_vis_observations = []
    actions = []
    rewards = []
    dones = []
    env_modes = []
    for i in range(NumOfAgent):
        vec_observations.append(0)
        vis_observations.append(0)
        next_vec_observations.append(0)
        next_vis_observations.append(0)
        actions.append(0)
        rewards.append(0)
        dones.append(False)
        env_modes.append(0)

    DQNAgent = DQNAgent()
    AgentsHelper.SendMessageToEnv("Env Connection Test")

    #run episodes!
    #Epsode Count increase when Agent0's episode is end
    totalStep = 0
    preLevelUpEpisodeCount = 0
    episodelosses = []
    vis_episodelosses = []
    episodeRewards = []
    for i in range(NumOfAgent):
        episodeRewards.append(0)

    #tracemalloc.start(10)
    for episodeCount in range(totalEpisodeCount):
        #time1 = tracemalloc.take_snapshot()
        dones[0] = False
        if episodeCount > trainEpisodeCount:
            train_mode = False
        
        while not dones[0]:
            totalStep +=1
            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
                vec_observation, vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
                vis_observation = vis_observation_list[0]
                vec_observations[behavior_name_Num] = vec_observation
                vis_observations[behavior_name_Num] = vis_observation
                if actionWith_visModelPredict == False:
                    action = DQNAgent.get_action_from_vector_observation(vec_observation, behavior_name)
                else:
                    action = DQNAgent.get_action_from_visual_observation(vis_observation, behavior_name)
                actionTuple = ConversionDataType.ConvertList2DiscreteAction(action,behavior_name)
                env.set_actions(behavior_name, actionTuple)
                actions[behavior_name_Num] = action

            env.step()

            for behavior_name in behavior_names:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                behavior_name_Num = ConversionDataType.ConvertBehaviorname2Num(behavior_name)
                next_vec_observation, next_vis_observation_list, done = AgentsHelper.getObservation(behavior_name)
                next_vis_observation = next_vis_observation_list[0]
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
                vis_loss = DQNAgent.train_vis_model(dones[0])
                episodelosses.append(loss)
                vis_episodelosses.append(vis_loss)

                if totalStep % (target_update_step) == 0:
                    DQNAgent.updateTarget()

        if episodeCount % save_episode_interval == 0 and episodeCount !=0:
            torch.save(DQNAgent.VecModel.state_dict(), save_path+"state_dict_model.pt")
            torch.save(DQNAgent.VisModel.state_dict(), save_path+"vis_state_dict_model.pt")
            print("Model saved..")

        if episodeCount % print_episode_interval == 0 and episodeCount !=0:
            print("episode:{}-step:{}//average loss = {:.3f}/vis_loss = {:.3f}, average rewards = {:.2f}|{:.2f}|{:.2f}, epsilon = {:.4f}".format(
                episodeCount,
                totalStep,
                np.mean(episodelosses),
                np.mean(vis_episodelosses), 
                np.mean(episodeRewards[0]), 
                np.mean(episodeRewards[1]), 
                np.mean(episodeRewards[2]), 
                DQNAgent.epsilon))
            episodelosses = []
            episodeRewards = []
            for i in range(NumOfAgent):
                episodeRewards.append(0)

        if episodeCount> levelUpEpisodeCount + preLevelUpEpisodeCount:
            for index, env_mode in enumerate(env_modes):
                if(env_mode) != max_env_level:
                    env_modes[index] += 1
            print("Env Level updated..")
            AgentsHelper.UpdateEnvLevel(env_modes)
            preLevelUpEpisodeCount = episodeCount

        #time2 = tracemalloc.take_snapshot()
        #stats = time2.compare_to(time1,'lineno')
        #for stat in stats[:3]:
            #print(stat)

    env.close()