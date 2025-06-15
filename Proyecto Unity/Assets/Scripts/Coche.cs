using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using TMPro;

public class Coche : Agent
{
    bool estado;

    [SerializeField] private Material vivo;
    [SerializeField] private Material muerto;
    [SerializeField] private MeshRenderer Carro;

    [SerializeField] private TMP_Text text;
    private bool manual = true;

    [SerializeField] private RayPerceptionSensorComponentBase sensorLaser;

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (Input.GetAxisRaw("Restart") > 0) EndEpisode();

        RayPerceptionOutput rayOutput = sensorLaser.RaySensor.RayPerceptionOutput;
        float L1 = rayOutput.RayOutputs[4].HitFraction;
        float L2 = rayOutput.RayOutputs[2].HitFraction;
        float L3 = rayOutput.RayOutputs[0].HitFraction;
        float L4 = rayOutput.RayOutputs[1].HitFraction;
        float L5 = rayOutput.RayOutputs[3].HitFraction;

        ActionSegment<float> contiuousActions = actionsOut.ContinuousActions;

        if (Input.GetAxisRaw("CamSwitch") > 0 || manual) 
        {
            manual = true;
            contiuousActions[0] = Input.GetAxisRaw("Vertical");
            contiuousActions[1] = Input.GetAxisRaw("Horizontal");
        }

        if (Input.GetAxisRaw("CamSwitch") < 0 || !manual)
        {
            manual = false;

            contiuousActions[0] = L3 + L2/2 + L4/2 - 1f;
            contiuousActions[1] = L5 + L4/2 - L2/2 - L1;
        }

        L1 = Mathf.Round(L1 * 10) * 0.1f;
        L2 = Mathf.Round(L2 * 10) * 0.1f;
        L3 = Mathf.Round(L3 * 10) * 0.1f;
        L4 = Mathf.Round(L4 * 10) * 0.1f;
        L5 = Mathf.Round(L5 * 10) * 0.1f;

        text.text = ("L1: " + L1 + "    L2: " + L2 + "    L3: " + L3 + "    L4: " + L4 + "    L5: " + L5);
        text.text += "\nVelocidad: " + Mathf.Round(contiuousActions[0] * 10) * 0.1 + "    Rotación: " + Mathf.Round(contiuousActions[1] * 10) * 0.1;

    }

    public override void Initialize()
    {
        estado = true;
        Carro.material = vivo;
    }

    public override void OnEpisodeBegin()
    {
        transform.localPosition = new Vector3(-90, 0.5f, -60);
        transform.localRotation = new Quaternion(0f, 0f, 0f, 0f);
        estado = true;
        Carro.material = vivo;
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
