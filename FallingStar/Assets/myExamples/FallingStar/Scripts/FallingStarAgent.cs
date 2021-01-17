using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class FallingStarAgent : Agent
{
    public GameObject star_pref;
    public GameObject Env;
    public GameObject Agent;

    public float star_speed;
    public float star_angle_random;
    public int starnum;
    public int Env_mode=0;

    private List<GameObject> stars = new List<GameObject>();


    public override void CollectObservations(VectorSensor sensor)
    {
        RaycastHit hit;
        float Angle;
        Ray ray;
        int rayCount = 27;
        List<Vector3> debugRay = new List<Vector3>();

        for (int i =0; i<=rayCount; i++)
        {
            Angle = i * Mathf.PI / rayCount;
            ray = new Ray(transform.position, new Vector3(0f, Mathf.Sin(Angle), Mathf.Cos(Angle)));
            if(Physics.Raycast(ray, out hit))
            {
                sensor.AddObservation(hit.distance);
                debugRay.Add(hit.point);
            }
        }
        for(int i = 0; i<debugRay.Count; i++)
        {
            Debug.DrawRay(transform.position, debugRay[i] - this.transform.position, Color.green);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var act0 = actionBuffers.DiscreteActions[0];
        var nextPosition = this.transform.position;
        var reward = 0f;
        switch (act0)
        {
            case 1:
                nextPosition += new Vector3(0f, 0f, 0.2f);
                break;
            case 2:
                nextPosition += new Vector3(0f, 0f, -0.2f);
                break;
            case 0:
                break;
        }

        Collider[] blockTest = Physics.OverlapBox(nextPosition, new Vector3(0.26f, 0.26f, 0.26f));

        if (blockTest.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length == 0)
        {
            this.transform.position = nextPosition;
        }
        else
        {
            reward += -0.1f;
        }
        
        blockTest = Physics.OverlapBox(this.transform.position, new Vector3(0.4f, 0.4f, 0.4f));
        if (blockTest.Where(col => col.gameObject.CompareTag("star")).ToArray().Length != 0)
        {
            reward = -1f;
            AddReward(reward);
            EndEpisode();
        }
        reward += 0.005f;
        AddReward(reward);
    }

    public override void OnEpisodeBegin()
    {
        this.transform.position = Env.transform.position;
        foreach (GameObject star in stars)
        {
            DestroyImmediate(star.gameObject);
        }
        stars.Clear();

        switch (Env_mode)
        {
            case 0:
                star_speed = 8f;
                starnum = 6;
                break;
            case 1:
                star_speed = 10f;
                starnum = 8;
                break;
            case 2:
                star_speed = 12f;
                starnum = 10;
                break;
        }

        for (int i = 0; i < starnum; i++)
        {
            GameObject star = Instantiate(star_pref, Env.transform);
            starScript script = star.GetComponent<starScript>();
            script.SetStar(Agent, Env, star_speed, star_angle_random);
            stars.Add(star);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var DiscreteActionsout = actionsOut.DiscreteActions;
        DiscreteActionsout[0] = 0;
        if(Input.GetKey(KeyCode.A))
        {
            DiscreteActionsout[0] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            DiscreteActionsout[0] = 2;
        }
    }
}
