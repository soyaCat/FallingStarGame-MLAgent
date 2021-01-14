from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import(
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid

class StringLogChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
    def on_message_received(self, msg : IncomingMessage):
        print(msg.read_string())
    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)

class ConversionDataType():
    def getEnvNum_from_behavior_name(self, behavior_name):
        EnvNum = 0
        behavior_name = str(behavior_name).split('?')[0]
        behavior_name = behavior_name[8:]
        EnvNum = int(behavior_name)
        return EnvNum

    def delete_last_char(self, message):
        message = message[:-1]
        return message


if __name__ == "__main__":
    string_log = StringLogChannel()
    ConversionDataType = ConversionDataType()
    game = "D:/program/unity/project/FallingStar/Build/FallingStar.exe"
    env = UnityEnvironment(file_name = game, side_channels = [string_log])
    env.reset()
    string_log.send_string("The environment was reset")
    behavior_names = list(env.behavior_specs)
    print(behavior_names)

    env_modes = [0,0,0]

    for i in range(1000):
        for behavior_name in behavior_names:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            behavior_name_Num = ConversionDataType.getEnvNum_from_behavior_name(behavior_name)
            #원래는 이 조건문에 에이전트의 상태를 가지고 behavior_name_num을 가지고  env_modes를 구분하여 모드를 업데이트 한다.
            if(i>500):
                env_modes[1] = 2

        sendMessage = ""
        for index, env_mode in enumerate(env_modes):
            sendMessage +=str(index) + "?" + str(env_mode)+"/"
        sendMessage = ConversionDataType.delete_last_char(sendMessage)
        string_log.send_string(
            "@" + sendMessage
        )
        env.step()

    env.close()