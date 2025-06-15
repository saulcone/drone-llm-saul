using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class Conducir : Agent
{
    bool estado;

    [SerializeField] private Material vivo1;
    [SerializeField] private Material vivo2;
    [SerializeField] private Material muerto;
    [SerializeField] private Material meta;
    [SerializeField] private MeshRenderer Carro;

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> contiuousActions = actionsOut.ContinuousActions;
        contiuousActions[0] = Input.GetAxisRaw("Vertical");
        contiuousActions[1] = Input.GetAxisRaw("Horizontal");
    }

    public override void Initialize()
    {
        estado = true;
        Carro.material = vivo1;
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(-90, 0.5f, -60);
        transform.localRotation = new Quaternion(0f, 0f, 0f, 0f);
        estado = true;
        Carro.material = vivo1;
        //if (gameObject.name.Length < 10) Carro.material = vivo2;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position);
        sensor.AddObservation(estado);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (estado) 
        {
            float velLineal = actions.ContinuousActions[0];
            float velAngular = actions.ContinuousActions[1];

            float moveSpeed = 25f;
            float rotateSpeed = 100f;
            transform.Rotate(Vector3.up, velAngular * rotateSpeed * Time.deltaTime, Space.Self);
            transform.localPosition += transform.forward * moveSpeed * Time.deltaTime * velLineal;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Paredes>(out Paredes paredes))
        {
            estado = false;
            Carro.material = muerto;
            print("Muerto");
        }
    }
}
