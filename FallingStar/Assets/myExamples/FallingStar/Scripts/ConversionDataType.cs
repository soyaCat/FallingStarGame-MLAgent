using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ConversionDataType
{
    public int[] MessagToIntList(string message)
    {
        char[] message_char = message.ToCharArray();
        if (message_char[0] == '@')
        {
            string[] splitedMessage = message.Split(new char[] { '@' });
            splitedMessage = splitedMessage[1].Split(new char[] { '/' });
            int[] env_modes = new int[splitedMessage.Length];
            foreach (string agentMessagesPack in splitedMessage)
            {
                char[] inner_message_char = agentMessagesPack.ToCharArray();
                env_modes[(int)inner_message_char[0] - 48] = (int)inner_message_char[2] - 48;
            }
            return env_modes;
        }
        else
        {
            return null;
        }
    }
}
