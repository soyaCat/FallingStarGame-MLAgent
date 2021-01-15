using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

using UnityEngine.UI;

public class RegisterStringLogSideChannel : MonoBehaviour
{
    public string message;
    public GameObject TextUI;
    public GameObject Env0Agent;
    public GameObject Env1Agent;
    public GameObject Env2Agent;

    private Text TextUIText;
    private FallingStarAgent Env0_Agentsc;
    private FallingStarAgent Env1_Agentsc;
    private FallingStarAgent Env2_Agentsc;

    StringLogSideChannel stringChannel;
    ConversionDataType conversionDataType;

    // Start is called before the first frame update
    public void Awake()
    {
        stringChannel = new StringLogSideChannel();
        conversionDataType = new ConversionDataType();
        Application.logMessageReceived += stringChannel.SendDebugStatementToPython;
        SideChannelManager.RegisterSideChannel(stringChannel);
    }

    public void OnDestroy()
    {
        Application.logMessageReceived -= stringChannel.SendDebugStatementToPython;
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(stringChannel);
        }
    }

    private void Start()
    {
        TextUIText = TextUI.GetComponent<Text>();
        Env0_Agentsc = Env0Agent.GetComponent<FallingStarAgent>();
        Env1_Agentsc = Env1Agent.GetComponent<FallingStarAgent>();
        Env2_Agentsc = Env2Agent.GetComponent<FallingStarAgent>();
    }
    

    // Update is called once per frame
    void Update()
    {
        this.message = stringChannel.message;
        if(this.message == null)
            this.message = "Receive Nothing!!";
        TextUIText.text = this.message;
        var env_modes = conversionDataType.MessagToIntList(this.message);
        if(env_modes != null)
        {
            Env0_Agentsc.Env_mode = env_modes[0];
            Env1_Agentsc.Env_mode = env_modes[1];
            Env2_Agentsc.Env_mode = env_modes[2];
        }
        else
        {
            TextUIText.text += "cant convert! message!";
        }
    }
}
