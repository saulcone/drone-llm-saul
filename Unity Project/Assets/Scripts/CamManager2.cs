using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CamManager2 : MonoBehaviour
{

    public GameObject cam1;
    public GameObject cam2;
    public GameObject cam3;

    private int cont;

    // Start is called before the first frame update
    void Start()
    {
        cam1.SetActive(true);
        cam2.SetActive(false);
        cam3.SetActive(false);
    }

    void SetCamera() 
    {
        if(cont == 0)
        {
            cam1.SetActive(true);
            cam2.SetActive(false);
            cam3.SetActive(false);
        }

        if (cont == 1)
        {
            cam1.SetActive(false);
            cam2.SetActive(true);
            cam3.SetActive(false);
        }

        if (cont == 2)
        {
            cam1.SetActive(false);
            cam2.SetActive(false);
            cam3.SetActive(true);
        }
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown("c")) cont++;
        if(Input.GetKeyDown("v")) cont--;

        cont = cont % 3;

        SetCamera();
    }
}
