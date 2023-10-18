#include "NoiseUtils.hlsl" 
float hh(in float n) { return frac(sin(n)*623.29091312);}
float noise(in float3 x) { float3 p = floor(x);float3 f = frac(x); f = f*f*(3.0 - 2.0*f); float n = p.x + p.y*157.0 + 113.0*p.z; return lerp(lerp(lerp(hh(n + 0.0), hh(n + 1.0), f.x), lerp(hh(n + 157.0), hh(n + 158.0), f.x), f.y), lerp(lerp(hh(n + 113.0), hh(n + 114.0), f.x),lerp(hh(n + 270.0), hh(n + 271.0), f.x), f.y), f.z); }




float snoise(float3 v)
{
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);

    // First corner
    float3 i  = floor(v + dot(v, C.yyy));
    float3 x0 = v   - i + dot(i, C.xxx);

    // Other corners
    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    // x1 = x0 - i1  + 1.0 * C.xxx;
    // x2 = x0 - i2  + 2.0 * C.xxx;
    // x3 = x0 - 1.0 + 3.0 * C.xxx;
    float3 x1 = x0 - i1 + C.xxx;
    float3 x2 = x0 - i2 + C.yyy;
    float3 x3 = x0 - 0.5;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    float4 p =
      permute(permute(permute(i.z + float4(0.0, i1.z, i2.z, 1.0))
                            + i.y + float4(0.0, i1.y, i2.y, 1.0))
                            + i.x + float4(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float4 j = p - 49.0 * floor(p / 49.0);  // mod(p,7*7)

    float4 x_ = floor(j / 7.0);
    float4 y_ = floor(j - 7.0 * x_);  // mod(j,N)

    float4 x = (x_ * 2.0 + 0.5) / 7.0 - 1.0;
    float4 y = (y_ * 2.0 + 0.5) / 7.0 - 1.0;

    float4 h = 1.0 - abs(x) - abs(y);

    float4 b0 = float4(x.xy, y.xy);
    float4 b1 = float4(x.zw, y.zw);

    //float4 s0 = float4(lessThan(b0, 0.0)) * 2.0 - 1.0;
    //float4 s1 = float4(lessThan(b1, 0.0)) * 2.0 - 1.0;
    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, 0.0);

    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    float3 g0 = float3(a0.xy, h.x);
    float3 g1 = float3(a0.zw, h.y);
    float3 g2 = float3(a1.xy, h.z);
    float3 g3 = float3(a1.zw, h.w);

    // Normalise gradients
    float4 norm = taylorInvSqrt(float4(dot(g0, g0), dot(g1, g1), dot(g2, g2), dot(g3, g3)));
    g0 *= norm.x;
    g1 *= norm.y;
    g2 *= norm.z;
    g3 *= norm.w;

    // Mix final noise value
    float4 m = max(0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    m = m * m;

    float4 px = float4(dot(x0, g0), dot(x1, g1), dot(x2, g2), dot(x3, g3));
    return 42.0 * dot(m, px);
}

float e3D(float3 input)
{
    float Out = snoise(input);
    return Out;
}



// This function calculates the intersection point of a ray with the cloud height boundaries
float IntersectCloudHeight(float3 rayStart, float3 rayDir, float4 cloudHeight)
{
    // Calculate the coefficients of the quadratic equation
    // The equation is derived from the dot product of the ray direction and the normal of the cloud height plane
    float a = rayDir.y * rayDir.y;
    float b = 2 * rayStart.y * rayDir.y;
    float c = rayStart.y * rayStart.y - cloudHeight.w * cloudHeight.w;

    // Calculate the discriminant of the quadratic equation
    // The discriminant determines if the ray intersects the cloud height plane or not
    float d = b * b - 4 * a * c;

    // If the discriminant is negative, there is no intersection
    if (d < 0)
    {
        return 0;
    }

    // If the discriminant is zero, there is one intersection
    if (d == 0)
    {
        return -b / (2 * a);
    }

    // If the discriminant is positive, there are two intersections
    // We need to find the closest one that is within the cloud height range
    float sqrtD = sqrt(d);
    float t1 = (-b - sqrtD) / (2 * a);
    float t2 = (-b + sqrtD) / (2 * a);

    // If both intersections are negative, there is no intersection
    if (t1 < 0 && t2 < 0)
    {
        return 0;
    }

    // If both intersections are positive, we need to find the smallest one
    if (t1 > 0 && t2 > 0)
    {
        return min(t1, t2);
    }

    // If one intersection is negative and one is positive, we need to find the positive one
    return max(t1, t2);
}
// This function calculates the lighting color of the clouds at a given position
// The color is based on the light direction, the view direction, the cloud brightness, the sun glare, the cloud scattering and the cloud height scatter parameters
float3 CloudLighting(float3 pos, float3 lightDir, float3 viewDir,
                     float cloudBrightness,
                     float sunGlare,
                     float cloudScattering,
                     float cloudHeightScatter, float3 plane,
                     float intensity) // bulut yoðunluðu
{
    // Calculate the half vector between the light direction and the view direction
    // The half vector is used to calculate the specular component of the lighting
    float3 halfDir = normalize(lightDir + viewDir);

    // Calculate the dot products of the light direction, the view direction and the half vector with the up vector
    // The up vector is assumed to be (0, 1, 0) in world space
    // These dot products are used to calculate the diffuse and specular components of the lighting
    float lightDot = saturate(dot(lightDir, plane));
    float viewDot = saturate(dot(viewDir, plane));
    float halfDot = saturate(dot(halfDir, plane));

    // Calculate the output color of the lighting
    // The output color is a combination of the cloud color, the diffuse, specular, scattering and height scatter components
    // The output color is clamped between 0 and 1
    float3 color = saturate(
        // Cloud color
        lerp(float3(1.0, 0.94, 0.81), float3(0.251, 0.31, 0.352), intensity) +
        // Diffuse component
        lightDot * cloudBrightness +
        // Specular component
        pow(halfDot, sunGlare) * cloudBrightness +
        // Scattering component
        pow(dot(lightDir, viewDir), cloudScattering) * cloudBrightness+
        // Height scatter component
        saturate(pos.y - cloudHeightScatter) * cloudHeightScatter +
        // Sun glare component
        sunGlare * float3(1.0, 0.22, 0.22) * pow(clamp(dot(lightDir, pos), 0.0, 1.0), 3.0)
    );

    // Return the output color of the lighting
    return color;
}

// This function samples a noise value at a given position
// The noise value is based on a 3D Perlin noise function
// The noise value is modulated by the cloud time and the cloud subtract parameters
float SampleNoise(float3 pos, float cloudTime, float cloudSubtract)
{
    // Calculate the noise frequency and amplitude
    // The frequency determines how fast the noise changes over space
    // The amplitude determines how much the noise affects the output value
    float frequency = 1;
    float amplitude = 1;

    // Initialize the noise value and the persistence factor
    // The persistence factor determines how fast the amplitude decreases for each octave of noise
    float n = 0;
    float persistence = 0.5;

    // Loop through four octaves of noise
    for (int i = 0; i < 4; i++)
    {
        // Sample the 3D Perlin noise value at the given position
        // The position is scaled by the frequency and offset by the cloud time
        // The Perlin noise value is in the range [-1, 1]
        float perlin = noise(pos * frequency);

        // Add the Perlin noise value to the output noise value
        // The Perlin noise value is scaled by the amplitude
        n += perlin * amplitude;

        // Decrease the amplitude for the next octave of noise
        // The amplitude is multiplied by the persistence factor
        amplitude *= persistence;

        // Increase the frequency for the next octave of noise
        // The frequency is doubled
        frequency *= 2;
    }

    // Subtract a constant value from the output noise value
    // This value is controlled by the cloud subtract parameter
    // This creates more contrast and variation in the output noise value
    n -= cloudSubtract;

    // Return the output noise value
    return n;
}
bool intersectPlane(in float3 n, in float3 p0, in float3 l0, in float3 l, out float t) 
		{ 
	    	float denom = dot(n, l);
	    	if (denom > 0) { 
	        	t = dot(p0 - l0, n) / denom;
	        	return (t >= 0); 
	  		} 
	    	return false; 
		} 


// Raymatch function that takes start and end positions, step count, cloud density, sharpness, opacity, sun angle and direction as inputs
// and returns a color value as output
float4 Raymatch(float3 start, float3 end, int steps, float density, float sharpness, float opacity, float sunAngle, float3 direction)
{
    // Initialize a color variable to store the result
  float3 color = float3(0.0, 0.0, 0.0);

   // Raymarch loop
   float dist = 0;
   float alpha = 0;
   float3 pos = start;
   float rayLength2 = distance(start, end);
   float stepsize2 = rayLength2 / steps;
   float shadow = 1; // Initialize the shadow value to 1
   float3 cloudColor = float3(0.0, 0.0, 0.0);
  // Loop from start to end position with the given step count
  for (int i = 0; i < steps; i++)
  {
    // Break if the ray is too far or the alpha is too high
    if (dist > rayLength2 || alpha > 0.99)
        break;

    // Calculate the cloud density at that position using a noise function
    float cloud = noise(pos * density);

    // Multiply the cloud density by the sharpness and opacity values
    cloud = saturate(pow(cloud, sharpness));

    // Accumulate the alpha based on the noise and the step size
    alpha += cloud * stepsize2 * opacity;

    // Calculate the normal vector using the gradient of the noise function
    float3 normal = normalize(float3(
    noise(pos * density + float3(0.01, 0, 0)) - noise(pos * density - float3(0.01, 0, 0)),
    noise(pos * density + float3(0, 0.01, 0)) - noise(pos * density - float3(0, 0.01, 0)),
    noise(pos * density + float3(0, 0, 0.01)) - noise(pos * density - float3(0, 0, 0.01))
    ));

    // Calculate the angle between the light vector and the normal vector
    float angle = acos(dot(sunAngle, normal));

    // Calculate the shadow value based on the angle
    shadow = 1 - (angle / 180);

    // Calculate the cloud color based on the sun angle and the cloud density
    cloudColor += lerp(float3(1, 1, 1), float3(1, 0.8, 0.6), saturate(angle / 90)) * lerp(0.8, 1.2, cloud);

    // Move the ray forward by the step size
    pos += direction * stepsize2;

    // Increase the total distance travelled
    dist += stepsize2;
  }
    // Mix the sky color and the cloud color based on the alpha and shadow values
    color = lerp(float3(0,0,0), cloudColor, alpha);

    // Return the result color
    return float4(color, alpha);
}

float4 rayCast(float3 start, float3 end, float3 direction,  float stepLength, float3 lDir, float2 hSpan, float3 skyColor, float light, float _CloudSubtract, float _CloudTime, float _CloudAlpha, float _CloudBrightness, float _CloudHardness, float _SunGlare, float _CloudScale) {

			float3 pos = start;
			float3 dir = normalize(direction);
			float intensity = 0;
			int N = stepLength;
			float sl = length(end - start)/N;
			float scale = 0.0001 * _CloudScale;
			float4 sum = float4(0,0,0,0);
            [loop]
			for (int i=0;i<N;i++)
				{
					if (pos.y >= hSpan.x*0.99 && pos.y <= hSpan.y && sum.a<0.99) {
						intensity = (SampleNoise(pos * _CloudScale, _CloudTime, _CloudSubtract))* _CloudAlpha;
					}
					else break;

					if (intensity > 0.01) { // Integrate
						float dif = clamp(intensity - SampleNoise(pos * _CloudScale, _CloudTime, _CloudSubtract)* 5,0,1);
						float3 l = float3(0.65,0.7,0.75)*_CloudBrightness + float3(1.0, 0.6, 0.8) * dif; 
	    				float4 c = float4(float3(1.0,0.94,0.81), intensity);

	    				c.xyz *= l;
	    				c.a *= _CloudHardness;
	    				c.rgb *= c.a;
                        float3 tempos = pos;
                        float tempintensity = 0;
          //              for(int j = 0; j<10; j++)
          //              {
          //                  if (tempos.y >= hSpan.x*0.99 && tempos.y <= hSpan.y && c.a < 0.99) 
          //                  {
						    //tempintensity = (SampleNoise(pos * _CloudScale, _CloudTime, _CloudSubtract))* _CloudAlpha;
          //                  tempos += tempos + (lDir*(-1)*sl);
          //                  //c *= .2;
					     //   }
          //                  else break;
          //              }
                        //c *= 1.2-SampleNoise((pos + (lDir*(-1)*sl)) * _CloudScale, _CloudTime, _CloudSubtract)*_CloudAlpha;
                        //c = lerp(c, float4(0.25,0.25,0.25,intensity), );
	    				sum = sum + c * (1.0 - sum.a);
                        
					}
					pos = pos + dir*sl;
			}
            sum.a = clamp(sum.a,0,1);
			float4 col = float4(1,1,1,1);
			col = sum;
			float sun = clamp( dot(dir,lDir), 0.0, 1.0 );
			col.a = sum.a;
            float4 sunc = _SunGlare* lerp(float4(1.0,0.15,0.15,1),float4(0,0,0,1) , sum.a);
            //col = lerp(sunc,col,pow(sun,5));
			//col.xyz *= light * skyColor;
			return col;
}


// This function calculates the color and transparency of the clouds
float4 CloudColor_float(float3 cameraPos, float3 viewDir,
                  float3 lightDir,
                  float4 cloudHeight,
                  float cloudScale,
                  float cloudDistance,
                  float maxDetail,
                  float cloudSubtract,
                  float cloudScattering,
                  float cloudAlpha,
                  float cloudHardness,
                  float cloudBrightness,
                  float sunGlare,
                  float cloudHeightScatter,
                  float cloudTime,
                  out half cR,
                  out half cG,
                  out half cB,
                  out half cA)
{
    // Initialize the output color and alpha values
    cR = 0;
    cG = 0;
    cB = 0;
    cA = 0;

    cloudScale*= .001;
    // Rayýn bulut katmanýyla kesiþip kesiþmediðini kontrol edelim
    // Eðer kesiþmiyorsa boþ renk döndürelim
    float t0, t1;
    float3 plane = float3(0,1,0);
    float3 plane1Pos = float3(0,cloudHeight.x,0);
    float3 plane2Pos = float3(0,cloudHeight.y,0);

    if (cameraPos.y < cloudHeight.x) { // Kamera bulut katmanýnýn altýndaysa
        if (intersectPlane(plane, plane1Pos, cameraPos, viewDir,t0)) {
            if (intersectPlane(plane, plane2Pos, cameraPos, viewDir,t1)) {
            }
            else discard;

        }
        else discard;
    }
    else discard;

    // Rayýn bulut katmanýyla kesiþtiði baþlangýç ve bitiþ noktalarýný hesaplayalým
    float3 startPos = cameraPos + t0*viewDir;
    float3 endPos = cameraPos + t1*viewDir;


    // Calculate the ray start and end positions in world space
    // The ray start is the camera position
    // The ray end is the point where the ray intersects the upper or lower boundary of the cloud height
    // The ray direction is normalized
    // The ray length is clamped by the cloud distance parameter
    float3 rayStart = cameraPos + t0*viewDir;
    float3 rayEnd = cameraPos + t1*viewDir;
    //rayEnd = min(rayEnd, rayStart + viewDir * cloudDistance);
    //viewDir = normalize(viewDir);
    float rayLength = distance(rayStart, rayEnd);

    // Calculate the number of steps for the ray marching algorithm
    // The number of steps is proportional to the ray length and inversely proportional to the cloud scale
    // The number of steps is clamped by the max detail parameter
    int steps = (int)ceil(rayLength * (1 / cloudScale));
    steps = min(steps, (int)maxDetail);

    // Calculate the step size for the ray marching algorithm
    // The step size is equal to the ray length divided by the number of steps
    // The step size is scaled by the cloud scale parameter
    float stepSize = rayLength / steps;
    stepSize *= cloudScale;

    // Initialize the ray position and the accumulated density
    float3 rayPos = rayStart;
    float density = 0;

    // Loop through the steps of the ray marching algorithm
    for (int i = 0; i < steps; i++)
    {
        // Sample the noise value at the current ray position
        // The noise value is a function of the ray position, the cloud scale, the cloud time and the cloud subtract parameters
        // The noise value is used to determine the cloud density at the current ray position
        float noise = SampleNoise(rayPos * cloudScale, cloudTime, cloudSubtract);


        // Calculate the cloud density at the current ray position
        // The cloud density is a function of the noise value, the cloud alpha and the cloud hardness parameters
        // The cloud density is clamped between 0 and 1
        float cloudDensity = saturate(noise * cloudAlpha * cloudHardness);

        // Accumulate the cloud density along the ray
        // The accumulated density is used to determine the opacity of the clouds
        density += cloudDensity * stepSize;

        // Calculate the color of the clouds at the current ray position
        // The color is a function of the light direction, the view direction, the cloud brightness, the sun glare, the cloud scattering and the cloud height scatter parameters
        // The color is modulated by the cloud density and added to the output color
        float3 color = CloudLighting(rayPos, lightDir, viewDir, cloudBrightness, sunGlare, cloudScattering, cloudHeightScatter, plane, density) * cloudDensity;
        cR += color.r;
        cG += color.g;
        cB += color.b;

        // Advance the ray position by the step size
        rayPos += viewDir * stepSize;
    }

    // Calculate the output alpha value
    // The alpha value is a function of the accumulated density and the cloud alpha parameter
    // The alpha value is clamped between 0 and 1
    cA = saturate(density * cloudAlpha);

    // Return the output color and alpha values as a float4 vector
    return float4(cR, cG, cB, cA);
}

// This function calculates the same thing as above but for half precision floats
half4 CloudColor_half(half3 cameraPos, half3 viewDir,
                  half3 lightDir,
                  half4 cloudHeight,
                  half cloudScale,
                  half cloudDistance,
                  half maxDetail,
                  half cloudSubtract,
                  half cloudScattering,
                  half cloudAlpha,
                  half cloudHardness,
                  half cloudBrightness,
                  half sunGlare,
                  half cloudHeightScatter,
                  float cloudTime,
                  out half cR,
                  out half cG,
                  out half cB,
                  out half cA)
{
    // Initialize the output color and alpha values
    cR = 0;
    cG = 0;
    cB = 0;
    cA = 0;
    
    cloudScale*= .001;
    // Rayýn bulut katmanýyla kesiþip kesiþmediðini kontrol edelim
    // Eðer kesiþmiyorsa boþ renk döndürelim
    float t0, t1;
    float3 plane = float3(0,1,0);
    float3 plane1Pos = float3(0,cloudHeight.x,0);
    float3 plane2Pos = float3(0,cloudHeight.y,0);

    if (cameraPos.y < cloudHeight.x) { // Kamera bulut katmanýnýn altýndaysa
        if (intersectPlane(plane, plane1Pos, cameraPos, viewDir,t0)) {
            if (intersectPlane(plane, plane2Pos, cameraPos, viewDir,t1)) {
            }
            else return half4(0, 0, 0, 0);
        }
        else return half4(0, 0, 0, 0);
    }
    else return half4(0, 0, 0, 0);


    // Calculate the ray start and end positions in world space
    // The ray start is the camera position
    // The ray end is the point where the ray intersects the upper or lower boundary of the cloud height
    // The ray direction is normalized
    // The ray length is clamped by the cloud distance parameter
    half3 rayStart = cameraPos + t0 * viewDir;
    half3 rayEnd = cameraPos + t1 * viewDir;
   // rayEnd = min(rayEnd, rayStart + viewDir * cloudDistance);
    //viewDir = normalize(viewDir);
    half rayLength = distance(rayStart, rayEnd);

    // Calculate the number of steps for the ray marching algorithm
    // The number of steps is proportional to the ray length and inversely proportional to the cloud scale
    // The number of steps is clamped by the max detail parameter
    int steps = (int)ceil(rayLength * (1 / cloudScale));
    steps = min(steps, (int)maxDetail);

    // Calculate the step size for the ray marching algorithm
    // The step size is equal to the ray length divided by the number of steps
    // The step size is scaled by the cloud scale parameter
    half stepSize = rayLength / steps;
    stepSize *= cloudScale;

    // Initialize the ray position and the accumulated density
    half3 rayPos = rayStart;
    half density = 0;

    // Loop through the steps of the ray marching algorithm
    for (int i = 0; i < steps; i++)
    {
        // Sample the noise value at the current ray position
        // The noise value is a function of the ray position, the cloud scale, the cloud time and the cloud subtract parameters
        //The noise value is        // used to determine the cloud density at the current ray position
        half noise = SampleNoise(rayPos * cloudScale, cloudTime, cloudSubtract);

        // Calculate the cloud density at the current ray position
        // The cloud density is a function of the noise value, the cloud alpha and the cloud hardness parameters
        // The cloud density is clamped between 0 and 1
        half cloudDensity = 0;
        if (rayPos.y >= cloudHeight.x*0.99 && rayPos.y <= cloudHeight.y && density < 2) {
	    	cloudDensity = saturate(e3D(rayPos * cloudScale) * cloudAlpha * cloudHardness);
            cloudDensity -= cloudSubtract;
        }
		else break;

        // Accumulate the cloud density along the ray
        // The accumulated density is used to determine the opacity of the clouds
        density += cloudDensity;

        // Calculate the color of the clouds at the current ray position
        // The color is a function of the light direction, the view direction, the cloud brightness, the sun glare, the cloud scattering and the cloud height scatter parameters
        // The color is modulated by the cloud density and added to the output color
        
        //half3 color = CloudLighting(rayPos, lightDir, viewDir, cloudBrightness, sunGlare, cloudScattering, cloudHeightScatter, plane, density) * cloudDensity;
        half angle = dot(lightDir, viewDir); // Vektörler arasýndaki açýnýn kosinüsü
        half3 color = lerp(float3(0.251, 0.31, 0.352), float3(1.0, 0.94, 0.81), angle * 0.5 +0.5) * density; // Açýya göre renk deðeri
        saturate(color);
        cR += color.r;
        cG += color.g;
        cB += color.b;

        // Advance the ray position by the step size
        rayPos += viewDir * 2;
    }

    //cR = 0;
    //cG = 0;
    //cB = 0;
    //cA = 0;

    half angle = dot(lightDir, viewDir); 

    float3 colorT;


    colorT = lerp(float3(0.251, 0.31, 0.352), float3(1.0, 0.94, 0.81), angle * 0.5 +0.5) * density;

    //colorT = Raymatch(rayStart,rayEnd,steps, 0.1,cloudHardness, cloudAlpha, lightDir, viewDir);
    cR = colorT.r;
    cG = colorT.g;
    cB = colorT.b;
    // Calculate the output alpha value
    // The alpha value is a function of the accumulated density and the cloud alpha parameter
    // The alpha value is clamped between 0 and 1
    cA = saturate(density);

    // Return the output color and alpha values as a half4 vector
    return half4(cR, cG, cB, cA);
}

