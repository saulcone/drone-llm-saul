using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class Pilot : Agent
{
    private bool isAlive;
    float animHelper = 3f;
    float forwardBackwardHelper = 0f;
    float leftRightHelper = 0f;
    private Transform droneVisualTransform;

    private int counter;

    [SerializeField] private RayPerceptionSensorComponentBase heightSensor;

    [SerializeField] private Material alive;
    [SerializeField] private Material finished;
    [SerializeField] private Material dead;
    [SerializeField] private MeshRenderer bodyRenderer;

    [SerializeField] public GameObject spawn;

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (Input.GetAxisRaw("Restart") > 0)
        {
            EndEpisode();
            Initialize();
        }
        if (Input.GetAxisRaw("Restart") < 0)
        {
            isAlive = false;
            bodyRenderer.material = dead;
            print("Dead");
        }

        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;

        continuousActions[0] = Input.GetAxisRaw("Dron1");
        continuousActions[1] = Input.GetAxisRaw("Dron2");
        continuousActions[2] = Input.GetAxisRaw("Dron3");
        continuousActions[3] = Input.GetAxisRaw("Dron4");
    }

    private void ResetDrone()
    {
        forwardBackwardHelper = 0f;
        leftRightHelper = 0f;
        transform.localPosition = spawn.transform.localPosition;
        transform.localRotation = spawn.transform.localRotation;
        isAlive = true;
        bodyRenderer.material = alive;

        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    public override void Initialize()
    {
        droneVisualTransform = gameObject.transform.GetChild(0);
        ResetDrone();
    }

    public override void OnEpisodeBegin()
    {
        ResetDrone();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.position);
        sensor.AddObservation(transform.rotation.eulerAngles.y);
        sensor.AddObservation(isAlive);
    }

    private void Animate(float upDown, float yaw, float forwardBackward, float leftRight)
    {
        if (forwardBackward == 0)
        {
            if (forwardBackwardHelper < 0) forwardBackwardHelper += 1;
            if (forwardBackwardHelper > 0) forwardBackwardHelper -= 1;
        }

        if (forwardBackward < 0) { if (forwardBackwardHelper < animHelper) forwardBackwardHelper += 1; }
        if (forwardBackward > 0) { if (forwardBackwardHelper > -animHelper) forwardBackwardHelper -= 1; }

        if (leftRight == 0)
        {
            if (leftRightHelper < 0) leftRightHelper += 1;
            if (leftRightHelper > 0) leftRightHelper -= 1;
        }

        if (leftRight > 0) { if (leftRightHelper < animHelper) leftRightHelper += 1; }
        if (leftRight < 0) { if (leftRightHelper > -animHelper) leftRightHelper -= 1; }

        droneVisualTransform.localRotation = Quaternion.Euler(forwardBackwardHelper, 180f, leftRightHelper);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (isAlive)
        {
            float moveSpeed = 5f;
            float rotateSpeed = 100f;

            float leftRight = actions.ContinuousActions[0];
            float forwardBackward = actions.ContinuousActions[1];
            float yaw = actions.ContinuousActions[2];
            float upDown = actions.ContinuousActions[3];

            Animate(upDown, yaw, forwardBackward, leftRight);

            transform.Rotate(Vector3.up, rotateSpeed * Time.deltaTime * yaw);
            transform.localPosition += transform.forward * moveSpeed * Time.deltaTime * forwardBackward;
            transform.localPosition += transform.right * moveSpeed * Time.deltaTime * leftRight;
            transform.localPosition += transform.up * moveSpeed * Time.deltaTime * upDown;
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        isAlive = false;
        bodyRenderer.material = dead;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Structure>(out Structure structure))
        {
            isAlive = false;
            bodyRenderer.material = dead;
        }

        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            isAlive = false;
            bodyRenderer.material = finished;
        }
    }
}
