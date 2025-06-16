using UnityEngine;
using Unity.MLAgents.SideChannels;
using System;

public class IntReceiverSideChannel : SideChannel
{
    public int receivedValue;

    public IntReceiverSideChannel()
    {
        ChannelId = new Guid("12345678-1234-1234-1234-1234567890ab");
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        receivedValue = msg.ReadInt32();
    }

    public int GetValue()
    {
        return receivedValue;
    }
}
