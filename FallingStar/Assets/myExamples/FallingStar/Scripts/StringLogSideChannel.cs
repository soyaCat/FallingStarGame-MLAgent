using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using System.Text;
using System;

public class StringLogSideChannel : SideChannel
{
    public string message;

    public StringLogSideChannel()
    {
        ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        var receivedString = msg.ReadString();
        message = receivedString;
    }

    public void SendDebugStatementToPython(string logString, string stackTrace, LogType type)
    {
        if(type == LogType.Error)
        {
            var stringtoSend = type.ToString() + ":" + logString + "\n" + stackTrace;
            using(var msgOut = new OutgoingMessage())
            {
                msgOut.WriteString(stringtoSend);
                QueueMessageToSend(msgOut);
            }
        }    
    }
}
