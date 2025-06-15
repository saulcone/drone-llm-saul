using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class Dron : Agent
{
    Vector3 InitialPos;
    Quaternion InitialRot;

    private bool estado;
    private int punt;
    [SerializeField] bool animacion;
    float anim_aux = 3f;
    float aux_fb = 0f;
    float aux_lr = 0f;
    private Transform DronPos;

    [SerializeField] private Camera camara;
    [SerializeField] private GameObject campos;

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
        contiuousActions[0] = Input.GetAxisRaw("Vertical");
        contiuousActions[1] = Input.GetAxisRaw("Horizontal");
        contiuousActions[2] = Input.GetAxisRaw("Dron");
        contiuousActions[3] = Input.GetAxisRaw("Dron2");
    }

    public override void Initialize()
    {
        InitialPos = transform.position;
        InitialRot = transform.rotation;

        estado = true;
        Carcasa.material = vivo1; 
        DronPos = this.gameObject.transform.GetChild(0);

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

    public void FijarCamara() 
    {
        camara.transform.position = campos.transform.position;
        camara.transform.rotation = campos.transform.rotation;
        camara.transform.SetParent(campos.transform);
    }

    public void Puntuar() 
    {
        punt++;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (estado)
        {
            float moveSpeed = 1f;
            float rotateSpeed = 100f;
            float ruido = 0.1f;

            float vud = actions.ContinuousActions[0];
            float vya = actions.ContinuousActions[1];
            float vfb = actions.ContinuousActions[2];
            float vlr = actions.ContinuousActions[3];

            if(ruido != 0) 
            {
                vud += Random.Range(-ruido, ruido);
                vya += Random.Range(-ruido, ruido);
                vfb += Random.Range(-ruido, ruido);
                vlr += Random.Range(-ruido, ruido);
            }

            float auxEstado = actions.ContinuousActions[4];

            if (animacion) Animar(vud, vya, vfb, vlr);

            transform.Rotate(Vector3.up, rotateSpeed * Time.deltaTime * vya);
            transform.localPosition += transform.forward * moveSpeed * Time.deltaTime * vfb;
            transform.localPosition += transform.right * moveSpeed * Time.deltaTime * vlr;
            transform.localPosition += transform.up * moveSpeed * Time.deltaTime * vud;

            if (auxEstado > 0.31)
            {
                Carcasa.material = final;
                estado = false;
            }
            else if (auxEstado > 0.21)
                Carcasa.material = vivo2;
            else if (auxEstado > 0.11)
                FijarCamara();
            else if (auxEstado > 0.01) 
            {
                Carcasa.material = muerto;
                estado = false;
            }  
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
