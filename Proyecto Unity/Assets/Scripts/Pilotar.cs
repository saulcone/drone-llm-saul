using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class Pilotar : Agent
{
    Vector3 InitialPos;
    Quaternion InitialRot;

    private bool estado;
    private int punt;
    float anim_aux = 3f;
    float aux_fb = 0f;
    float aux_lr = 0f;
    private Transform DronPos;

    private int cont;

    [SerializeField] private RayPerceptionSensorComponentBase sensorAltura;

    [SerializeField] private Material vivo1;
    [SerializeField] private Material vivo2;
    [SerializeField] private Material final;
    [SerializeField] private Material muerto;
    [SerializeField] private MeshRenderer Carcasa;

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if(Input.GetAxisRaw("Restart") > 0) EndEpisode();
        if (Input.GetAxisRaw("Restart") < 0)
        {
            estado = false;
            Carcasa.material = muerto;
            print("Muerto");
        }

        ActionSegment<float> contiuousActions = actionsOut.ContinuousActions;

        contiuousActions[0] = Input.GetAxisRaw("Dron4");
        contiuousActions[1] = Input.GetAxisRaw("Dron3");
        contiuousActions[2] = Input.GetAxisRaw("Dron2");
        contiuousActions[3] = Input.GetAxisRaw("Dron1");

    }

    public override void Initialize()
    {
        InitialPos = transform.position;
        InitialRot = transform.rotation;

        estado = true;
        Carcasa.material = vivo1; 
        DronPos = gameObject.transform.GetChild(0);

        punt = 0;
    }

    public override void OnEpisodeBegin()
    {
        aux_fb = 0f;
        aux_lr = 0f;
        transform.localPosition = InitialPos;
        transform.localRotation = InitialRot;
        estado = true;
        Carcasa.material = vivo1;

        punt = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position);
        sensor.AddObservation(estado);
    }

    public void Puntuar()
    {
        punt += 1;
        //print(punt);
    }

    private void Animar(float vud, float vya, float vfb, float vlr)
    {
        if (vfb == 0) 
        {
            if (aux_fb < 0) aux_fb += 1;
            if (aux_fb > 0) aux_fb -= 1;
        }

        if (vfb < 0) { if (aux_fb < anim_aux) aux_fb += 1; }
        if (vfb > 0) { if (aux_fb > -anim_aux) aux_fb -= 1; }

        if (vlr == 0)
        {
            if (aux_lr < 0) aux_lr += 1;
            if (aux_lr > 0) aux_lr -= 1;
        }

        if (vlr > 0) { if (aux_lr < anim_aux) aux_lr += 1; }
        if (vlr < 0) { if (aux_lr > -anim_aux) aux_lr -= 1; }

        DronPos.localRotation = Quaternion.Euler(aux_fb, 180f, aux_lr);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (estado)
        {
            float moveSpeed = 5f;
            float rotateSpeed = 100f;

            float vud = actions.ContinuousActions[0];
            float vya = actions.ContinuousActions[1];
            float vfb = actions.ContinuousActions[2];
            float vlr = actions.ContinuousActions[3];

            Animar(vud, vya, vfb, vlr);

            transform.Rotate(Vector3.up, rotateSpeed * Time.deltaTime * vya);
            transform.localPosition += transform.forward * moveSpeed * Time.deltaTime * vfb;
            transform.localPosition += transform.right * moveSpeed * Time.deltaTime * vlr;
            transform.localPosition += transform.up * moveSpeed * Time.deltaTime * vud;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Estructura>(out Estructura estructura))
        {
            estado = false;
            Carcasa.material = muerto;
            print("Muerto");
        }

        if (other.TryGetComponent<Meta>(out Meta meta))
        {
            estado = false;
            Carcasa.material = final;
            print("Meta");
        }
    }
}
