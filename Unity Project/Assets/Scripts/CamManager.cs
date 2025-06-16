using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CamManager : MonoBehaviour
{

    public GameObject cam1;
    public GameObject cam2;

    // Start is called before the first frame update
    void Start()
    {
        cam1.SetActive(true);
        cam2.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetAxis("CamSwitch") > 0) 
        {
            cam1.SetActive(true);
            cam2.SetActive(false);
        }
        if (Input.GetAxis("CamSwitch") < 0)
        {
            cam2.SetActive(true);
            cam1.SetActive(false);
        }
    }
}
