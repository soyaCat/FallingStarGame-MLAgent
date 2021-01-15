using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class starScript : MonoBehaviour
{
    GameObject Agent;
    GameObject Env;
    float star_speed;
    float star_angle_random;

    public void SetStar(GameObject Agent,GameObject Env, float star_speed, float star_angle_random)
    {
        this.Agent = Agent;
        this.Env = Env;
        this.star_speed = star_speed;
        this.star_angle_random = star_angle_random;

        RandomStar();
    }

    public void RandomStar()
    {
        Vector3 setPoint = new Vector3(0f, 17f, Random.Range(-15f, 15f));
        this.transform.localPosition = setPoint;
        Rigidbody rig = this.GetComponent<Rigidbody>();

        float randAngle = Mathf.Atan2(
            (Agent.transform.localPosition.z - this.transform.localPosition.z),
            (Agent.transform.localPosition.y - this.transform.localPosition.y))
            + Random.Range(-star_angle_random, star_angle_random);
        float randSpeed = star_speed + Random.Range(-0.5f, 0.5f);
        rig.velocity = new Vector3(0f, randSpeed * Mathf.Sin(randAngle), randSpeed * Mathf.Cos(randAngle));
    }

    private void OnCollisionEnter(Collision col)
    {
        if(col.gameObject.CompareTag("floor"))
        {
            RandomStar();
        }
    }
}
