using UnityEngine;
using static UnityEngine.InputSystem.LowLevel.InputStateHistory;
using static UnityEngine.Rendering.DebugUI;

public class DayNightCycle : MonoBehaviour
{
    public float dayLength = 300;
    public float passedTime = 20;

    [VectorLabels("Hour","Min","Sec")]public Vector3 gameTime = Vector3.zero;

    [SerializeField] Light sunLight;
    [SerializeField] Light moonLight;

    [SerializeField] Material skyboxMaterial;

    [SerializeField] Vector2 sunClamp = new Vector2(0,0.15f);
    [SerializeField] Vector2 moonClamp = new Vector2(0,0.15f);
    [SerializeField] Vector2 ambientClamp = new Vector2(-0.45f,-0.1f);

    [SerializeField][ColorUsage(false,true)] Color32 ambientColorDay = Color.white;
    [SerializeField][ColorUsage(false, true)] Color32 ambientColorNight = Color.gray;

    [SerializeField] float sunIntensityMax = 1f;
    [SerializeField] float moonIntensityMax = .5f;

    float dayRate;
    public Vector3 sunAngle;

    void Start()
    {
        dayRate = 86400f / dayLength;
    }
    void Update()
    {
        passedTime += Time.deltaTime;
        if (passedTime >= dayLength)
        {
            passedTime -= dayLength;
        }
        float rotation = 360 * (passedTime / dayLength);

        sunLight.transform.localEulerAngles = new Vector3(rotation - 90,-30,0);
        moonLight.transform.rotation = Quaternion.Euler(rotation + 90,-30,0);

        gameTime = new Vector3((int)(passedTime * dayRate / 3600f), (int)((passedTime * dayRate % 3600f) / 60f), (int)(passedTime * dayRate % 60f));

        sunAngle = sunLight.transform.forward.normalized;
        skyboxMaterial.SetVector("_SunAngle", sunAngle);
        
        Color ambientResult = Color.Lerp(ambientColorDay, ambientColorNight, CalculatedValue(sunAngle.y, ambientClamp)); // smoothstep uygula
        RenderSettings.ambientSkyColor = ambientResult * (.5f + (Mathf.Abs(sunAngle.y) * .25f));
        RenderSettings.ambientEquatorColor = ambientResult * (1f - (Mathf.Abs(sunAngle.y) * .25f));

        if(rotation <= 270 && rotation >= 90)
        {
            moonLight.intensity = 0;
            float sunResult = Mathf.Lerp(0, sunIntensityMax, CalculatedValue(-sunAngle.y, sunClamp));
            sunLight.intensity = sunResult;
        }
        else
        {
            sunLight.intensity = 0;
            float moonResult = Mathf.Lerp(0, moonIntensityMax, CalculatedValue(sunAngle.y,moonClamp));
            moonLight.intensity = moonResult;
        }
    }
    float CalculatedValue(float vector, Vector2 clamp)
    {
        float value = Mathf.Clamp(vector, clamp.x, clamp.y); // value deðiþkenini -0.25 ile 0.25 arasýna sýnýrla
        value = (value + Mathf.Abs(clamp.x)) * (1 / (clamp.y - clamp.x)); // value deðiþkenini 0 ile 1 arasýna dönüþtür
        return value;
    }
}


