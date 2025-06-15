using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnRandom : MonoBehaviour
{
    [SerializeField] public GameObject spawnsList;
    [SerializeField] public GameObject spawn;

    public void SetRandomSpawn()
    {
        int num_childs = spawnsList.transform.childCount;

        int random_child = UnityEngine.Random.Range(0, num_childs);

        Transform child_trasform = spawnsList.transform.GetChild(random_child);

        Debug.Log(child_trasform.position);
        Debug.Log(child_trasform.rotation);

        spawn.transform.SetPositionAndRotation(child_trasform.position, child_trasform.rotation);
    }
}
