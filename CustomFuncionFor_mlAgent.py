from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.base_env import ActionTuple
import uuid
import random
import numpy as np

class StringLogChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
    def on_message_received(self, msg : IncomingMessage):
        print(msg.read_string())

class ConversionDataType:
    def ConvertBehaviorname2Num(self, behavior_name):
        EnvNum = 0
        behavior_name = str(behavior_name).split('?')[0]
        behavior_name = behavior_name[8:]
        EnvNum = int(behavior_name)
        return EnvNum

    def ConvertList2DiscreteAction(self, arr, behavior_name):
        '''
        input data type = list -> ex)[3]
                !!! Dont Input 2D Array or list like [(0,2)], just input like [0,2]
        output data type = Actiontuple
        '''
        actionList = []
        actionList.append(arr)
        _discrete = np.array(actionList, dtype=np.int32)
        action = ActionTuple(discrete=_discrete)

        return action


    def delete_last_char(self, message):
        message = message[:-1]
        return message

class AgentsHelper:
    def __init__(self, Env, string_log):
        self.env = Env
        self.string_log = string_log
    def print_specs_of_Agents(self, behavior_names):
        for behavior_name in behavior_names:
            spec = self.env.behavior_specs[behavior_name]
            print(f"Name of the behavior : {behavior_name}")
            print("Number of observations : ", len(spec.observation_shapes))
            print("Observation shape : ", spec.observation_shapes)
            vis_obs_bool = any(len(shape) == 3 for shape in spec.observation_shapes)
            print("Is there a visual observation ?", vis_obs_bool)
            print("Is action is discrete ?", spec.action_spec.is_discrete())
            print("Is action is continus ?", spec.action_spec.is_continuous())
            print("\n")
        print("Examine finish....")
        print("======================================")

    def getObservation(self, behavior_name): 
        '''
        output data shape(visual_observation):
        -> (num_of_vis_obs_per_behavior_name, vis_obs_width, vis_obs_height, vis_obs_channel*stacked_data_num)
        output data shape(vector_observation):
        -> (1, num_of_vec_obs_per_behavior_name*stacked_data_num)

        output datatype is numpy array

        if terminal_steps.observations are exist, They overWrite decision_steps.observation
        '''
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        spec = self.env.behavior_specs[behavior_name]
        done = False
        vis_obs = []
        vec_obs = []
        tr_vis_obs = []
        tr_vec_obs = []
        for index, shape in enumerate(spec.observation_shapes):
            if len(shape) == 3:
                vis_obs.append(decision_steps.obs[index])
                tr_vis_obs.append(terminal_steps.obs[index])
        for index, shape in enumerate(spec.observation_shapes):
            if len(shape) == 1:
                vec_obs.append(decision_steps.obs[index])
                tr_vec_obs.append(terminal_steps.obs[index])

        if(tr_vec_obs[0].size != 0):
            vec_obs = tr_vec_obs
            done = True
        if(tr_vis_obs[0].size != 0):
            vis_obs = tr_vis_obs
            done = True
        vec_obs_num = len(vec_obs)
        vis_obs_num = len(vis_obs)
        vis_obs_shape = np.shape(vis_obs[0])

        for index, vec_obs_Ele in enumerate(vec_obs):
            vec_obs[index] = vec_obs_Ele.flatten()
        for index, vis_obs_Ele in enumerate(vis_obs):
            vis_obs[index] = vis_obs_Ele.flatten()
        vec_observation = np.array(vec_obs)
        vis_observation = np.uint8(255*np.array(vis_obs))
        vec_observation = vec_observation.reshape((vec_obs_num, -1))
        vis_observation = vis_observation.reshape((vis_obs_num, vis_obs_shape[1],vis_obs_shape[2],vis_obs_shape[3]))

        return vec_observation, vis_observation, done

    def getObservation_lite(self, behavior_name): 
        '''
        output data shape(visual_observation):
        -> (num_of_vis_obs_per_behavior_name, vis_obs_width, vis_obs_height, vis_obs_channel*stacked_data_num)
        output data shape(vector_observation):
        -> (1, num_of_vec_obs_per_behavior_name*stacked_data_num)

        output datatype is numpy array

        if terminal_steps.observations are exist, They overWrite decision_steps.observation
        use this function if vis_observation count is 1 per agent
        use less memory than getObservation()
        '''
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        spec = self.env.behavior_specs[behavior_name]
        done = False
        vis_obs = 0
        tr_vis_obs = 0
        vec_obs = 0
        tr_vec_obs = 0
        for index, shape in enumerate(spec.observation_shapes):
            if len(shape) == 3:
                vis_obs = (decision_steps.obs[index])
                tr_vis_obs = (terminal_steps.obs[index])
        for index, shape in enumerate(spec.observation_shapes):
            if len(shape) == 1:
                vec_obs = (decision_steps.obs[index])
                tr_vec_obs = (terminal_steps.obs[index])
        
        if(tr_vec_obs.size != 0):
            vec_obs = tr_vec_obs
            done = True
        if(tr_vis_obs.size != 0):
            vis_obs = tr_vis_obs
            done = True
        vis_obs_shape = np.shape(vis_obs)
        vec_obs = np.array(vec_obs)
        vis_obs = np.uint8(255*np.array(vis_obs))
        vec_obs = vec_obs.reshape((1, -1))
        vis_obs = vis_obs.reshape((1, vis_obs_shape[1],vis_obs_shape[2],vis_obs_shape[3]))

        return vec_obs, vis_obs, done


    def get_reward(self, behavior_name):
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        reward = decision_steps.reward
        tr_reward = terminal_steps.reward

        if(np.size(tr_reward)!=0):
            reward = tr_reward
        return reward[0]

    def UpdateEnvLevel(self, env_modes):
        sendMessage = ""
        print(env_modes)
        for index, env_mode in enumerate(env_modes):
            sendMessage +=str(index) + "?" + str(env_mode)+"/"
        sendMessage = ConversionDataType.delete_last_char(sendMessage)
        self.string_log.send_string("@" + sendMessage)
        self.env.step()

    def SendMessageToEnv(self, message):
        self.string_log.send_string(message)
        self.env.step()

    def sliceVisualObservation_ChannelLevel(self, vis_obs, stacked_data_num):
        '''
        input shape: (width, height, channel*stacked_data_num)
        output shape: (stacked_data_num, width, height, channel)

        input datatype is numpy array
        output datatype is list with numpy array
        '''
        vis_obs_shape = np.shape(vis_obs)
        vis_obs_list = []
        if(int(vis_obs_shape[2]/stacked_data_num)==3):
            for i in range(int(vis_obs_shape[2]/stacked_data_num)):
                vis_obs_list.append(vis_obs[:,:,i*3:(i+1)*3])
        if(int(vis_obs_shape[2]/stacked_data_num)==1):
            for i in range(int(vis_obs_shape[2]/stacked_data_num)):
                vis_obs_list.append(vis_obs[:,:,i:(i+1)])
        
        return vis_obs_list

    def saveArrayAsImagefile(self, array):
        '''
        input shape: (width, height, 3)
        '''
        im = Image.fromarray(array)
        im.save("your_file"+str(random.random())+".jpeg")
