using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Puntos : MonoBehaviour
{
    List<int> drones = new List<int>();

    void Start()
    {
        drones.Clear();
    }

    private void OnTriggerEnter(Collider other)
    {
        int id = other.GetInstanceID();
        if (!drones.Contains(id)) 
        {
            if (other.TryGetComponent<Pilotar>(out Pilotar pilotar))
            {
                pilotar.Puntuar();
            }

            if (other.TryGetComponent<Dron>(out Dron dron))
            {
                dron.Puntuar();
            }

            drones.Add(id);
        }
    }
}
