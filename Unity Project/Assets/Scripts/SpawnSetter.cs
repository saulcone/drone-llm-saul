using System.Collections;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class SpawnSetter : MonoBehaviour
{
    [SerializeField] public GameObject spawn;
    [SerializeField] public GameObject drone;

    IntReceiverSideChannel intChannel;

    public void Awake()
    {
        intChannel = new IntReceiverSideChannel();
        SideChannelManager.RegisterSideChannel(intChannel);
    }

    public void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            SideChannelManager.UnregisterSideChannel(intChannel);
        }
    }

    void Start()
    {
        StartCoroutine(WaitAndSetSpawn());
    }

    IEnumerator WaitAndSetSpawn()
    {
        yield return new WaitForSeconds(1f);
        int index = intChannel.GetValue();
        Debug.Log("Spawn index received: " + index);
        SetSpawn(index);
    }

    public void SetSpawn(int index)
    {
        if (index >= 0 && index < transform.childCount)
        {
            Transform childTransform = transform.GetChild(index);
            spawn.transform.SetPositionAndRotation(childTransform.position, childTransform.rotation);
            drone.transform.SetPositionAndRotation(childTransform.position, childTransform.rotation);
        }
        else
        {
            Debug.LogWarning("Invalid spawn index received: " + index);
        }
    }
}
