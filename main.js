 import { Stats } from "./stats.js";
 // Get the canvas and context
 var canvas = document.getElementById("canvas");
 canvas.width = window.innerWidth;
 canvas.height = window.innerHeight;
 var gl = canvas.getContext("webgl2");
 var ext = gl.getExtension('EXT_color_buffer_half_float');

 // Create the shaders
 var shaderSources = {
     vertex: `#version 300 es
        precision highp float;
        in vec2 aPosition;
        out vec2 vPosition;
        void main() {
            vPosition = aPosition;
            gl_Position = vec4(aPosition, 0, 1);
        }
    `,
     fragmentMain: /*glsl*/ `#version 300 es
     precision highp float;

     // Define shader model 3.0
     layout(location = 0) out vec4 pc_FragColor;
     uniform sampler2D colorTexture;
    uniform sampler2D heightTexture;
    uniform vec3 cameraPos;
    uniform vec3 cameraDir;
    uniform highp float time;
   in vec2 vPosition;
   
   const float sphereRadius = 0.5;
   const vec3 skyColor = vec3(0.5, 0.7, 1.0);
   const vec3 groundColor = vec3(0.4, 0.2, 0.1);
   const vec3 sphereColor = vec3(1.0, 0.3, 0.1);
   const float ambient = 0.1;
   
   float scene(vec3 pos) {
       float sphereDist = length(pos - vec3(0.0, sphereRadius + 0.01, 2.0)) - sphereRadius;
       float planeDist = pos.y;
       return min(sphereDist, planeDist);
   }
   
   float rayMarch(vec3 ro, vec3 rd) {
       float t = 0.0;
       for(int i = 0; i < 2048; i++) {
           float res = scene(ro + rd * t);
           if(res < 0.001 || t > 1000.0) return t;
           t += res;
       }
       return -1.0;
   }
   
   vec3 getNormal(vec3 pos) {
       vec2 e = vec2(0.001, 0.0);
       return normalize(vec3(
           scene(pos + e.xyy) - scene(pos - e.xyy),
           scene(pos + e.yxy) - scene(pos - e.yxy),
           scene(pos + e.yyx) - scene(pos - e.yyx)
       ));
   }
   
   mat4 makeViewMatrix(vec3 eye, vec3 center, vec3 up) {
       vec3 f = normalize(center - eye);
       vec3 s = normalize(cross(f, up));
       vec3 u = cross(s, f);
       return mat4(
           vec4(s, 0.0),
           vec4(u, 0.0),
           vec4(-f, 0.0),
           vec4(0.0, 0.0, 0.0, 1)
       );
   }
   vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
       vec2 xy = fragCoord - size / 2.0;
       float z = size.y / tan(radians(fieldOfView) / 2.0);
       return normalize(vec3(xy, -z));
   }
   vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
   vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
   
   float snoise(vec3 v){ 
     const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
     const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
   
   // First corner
     vec3 i  = floor(v + dot(v, C.yyy) );
     vec3 x0 =   v - i + dot(i, C.xxx) ;
   
   // Other corners
     vec3 g = step(x0.yzx, x0.xyz);
     vec3 l = 1.0 - g;
     vec3 i1 = min( g.xyz, l.zxy );
     vec3 i2 = max( g.xyz, l.zxy );
   
     //  x0 = x0 - 0. + 0.0 * C 
     vec3 x1 = x0 - i1 + 1.0 * C.xxx;
     vec3 x2 = x0 - i2 + 2.0 * C.xxx;
     vec3 x3 = x0 - 1. + 3.0 * C.xxx;
   
   // Permutations
     i = mod(i, 289.0 ); 
     vec4 p = permute( permute( permute( 
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
   
   // Gradients
   // ( N*N points uniformly over a square, mapped onto an octahedron.)
     float n_ = 1.0/7.0; // N=7
     vec3  ns = n_ * D.wyz - D.xzx;
   
     vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
   
     vec4 x_ = floor(j * ns.z);
     vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
   
     vec4 x = x_ *ns.x + ns.yyyy;
     vec4 y = y_ *ns.x + ns.yyyy;
     vec4 h = 1.0 - abs(x) - abs(y);
   
     vec4 b0 = vec4( x.xy, y.xy );
     vec4 b1 = vec4( x.zw, y.zw );
   
     vec4 s0 = floor(b0)*2.0 + 1.0;
     vec4 s1 = floor(b1)*2.0 + 1.0;
     vec4 sh = -step(h, vec4(0.0));
   
     vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
     vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
   
     vec3 p0 = vec3(a0.xy,h.x);
     vec3 p1 = vec3(a0.zw,h.y);
     vec3 p2 = vec3(a1.xy,h.z);
     vec3 p3 = vec3(a1.zw,h.w);
   
   //Normalise gradients
     vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
     p0 *= norm.x;
     p1 *= norm.y;
     p2 *= norm.z;
     p3 *= norm.w;
   
   // Mix final noise value
     vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
     m = m * m;
     return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                   dot(p2,x2), dot(p3,x3) ) );
   }
   #define NUM_OCTAVES 8

    float fbm(vec3 x) {
        float v = 0.0;
        float a = 0.5;
        vec3 shift = vec3(100);
        for (int i = 0; i < NUM_OCTAVES; ++i) {
            v += a * snoise(x);
            x = x * 2.0 + shift;
            a *= 0.75;
        }
        return v;
    }
   float traceHeightMap(vec3 ro, vec3 rd, float heightMagnitude, float minT) {
         vec3 samplePos = ro;
            bool hit = false;
            if (ro.y > heightMagnitude && rd.y < 0.0) {
                minT = (ro.y - heightMagnitude) / -rd.y;
            }
            float maxT = 1000.0;
            float dt = 0.01;
            float lh = 0.0;
            float ly = 0.0;
            float resT = -1.0;
            float t = minT;
            float maxIters = 256.0;
            while(t < maxT) {
                vec3 p = ro + rd * t;
                float h = heightMagnitude * textureLod(heightTexture, (p.xz * 0.5 + 0.5) * 0.01 + vec2(0.5), 0.0).r;

                if (p.y < h) {
                    resT = t - dt + dt * (lh-ly) / (p.y-ly-h+lh);
                    hit = true;
                    break;
                }

                dt = max(0.4 * abs(p.y - h), 0.001);//max(clamp(0.01 * abs(p.y - h), 0.01, 0.1) * t, 0.01);
                lh = h;
                ly = p.y;
                t += dt;
            }
           
            // Binary search to clean up
            if (hit) {
                for(int i = 0; i < 8; i++) {
                    dt *= 0.5;
                    vec3 p = ro + rd * resT;
                    float h = heightMagnitude * textureLod(heightTexture, (p.xz * 0.5 + 0.5) * 0.01 + vec2(0.5), 0.0).r;
                    if (p.y < h) {
                        resT -= dt;
                    } else {
                        resT += dt;
                    }

                }

            }

            return resT;
    }
    float traceShadowRay(vec3 ro, vec3 rd, float heightMagnitude, float minT) {
        float res = 1.0;
        float t = minT;
        float total = 0.0;
        for (int i = 0; i < 80; i++) {
            vec3 pos = ro + rd * t;
            float h = heightMagnitude * textureLod(heightTexture, (pos.xz * 0.5 + 0.5) * 0.01 + vec2(0.5), 2.0).r;
            float hei = pos.y - h + 0.25;
            res = min(res, 8.0 * hei / t);
            if (hei < 0.0) {
                total += t;
            }
            if (res < 0.0001 || pos.y > heightMagnitude) {
                break;
            }
            t += clamp( hei, 0.5+t*0.05, 25.0 );
        }
        return clamp(res, 0.0, 1.0);
    }        
    float smin(float a, float b, float k) {
        float h = max(k - abs(a-b), 0.) / k;
        return min(a, b) - h*h*h*k*1./6.;
    }
    float getCloudDensity(vec3 pos, float cloudPlaneHeight, float cloudPlaneRange) {
        float cloudSample = textureLod(heightTexture,  0.0025 * pos.xz - time * 0.01, 0.0).r;
        cloudSample = cloudSample * cloudSample;
        cloudSample *= pow(
            1.0 -  clamp(abs(pos.y - cloudPlaneHeight) / cloudPlaneRange, 0.0, 1.0), 
            5.0 - 4.0 * cloudSample
        );
        return cloudSample;
    }
    float rand(vec2 n) { 
        return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
    }
    vec4 computeFogColor( float t, vec3 ro, vec3 rd, vec3 lightDir ) {
        vec3 fd = rd;
        if (fd.y == 0.0) fd.y = 0.0001;
        float a = 0.1;
        float b = 0.15;
        float sunAmount = max( dot( rd, lightDir ), 0.0 );
        vec3 fogColor =  mix( vec3(0.5,0.6,0.7), // bluish
        vec3(1.0,0.9,0.7), // yellowish
        clamp(pow(sunAmount,8.0), 0.0, 1.0) );
        float fogAmount = clamp((a/b) * exp(-ro.y*b) * (1.0-exp( -t*fd.y*b ))/fd.y, 0.0, 1.0);
        vec4 fog = vec4( fogColor, fogAmount );
        return fog;
    }
    vec3 applyClouds(vec3 color, vec3 ro, vec3 rd, float t, vec3 lightDir, bool justIntensity) {
        float cloudPlaneHeight = 25.0;
        float cloudPlaneRange = 10.0;
        float minT = 0.0;
        float maxT = t;

        float bottomT = (ro.y - (cloudPlaneHeight - cloudPlaneRange)) / -rd.y;
        float topT = (ro.y - (cloudPlaneHeight + cloudPlaneRange)) / -rd.y;

        float enterT = min(bottomT, topT);
        float exitT = max(bottomT, topT);

        if (enterT > minT) {
            minT = enterT;
        }
        if (exitT < maxT) {
            maxT = exitT;
        }
        if (maxT > 1000.0) {
            maxT = 1000.0;
        }
        float totalIntensity = 0.0;
        float incidentLight = 1.0;
        float samples = 32.0;
        vec3 integratedColor = vec3(0.0);
        float integratedWeight = 0.0;
        for(float i = 0.0; i < samples; i++) {
            float cloudT = mix(minT, maxT, (i + rand(gl_FragCoord.xy)) / samples);
            vec3 cloudPos = ro + rd * cloudT;
            float cloudIntensity = getCloudDensity(cloudPos, cloudPlaneHeight, cloudPlaneRange);
            float cloudShadow = getCloudDensity(cloudPos + lightDir * 0.1, cloudPlaneHeight, cloudPlaneRange);
            totalIntensity += cloudIntensity * (maxT - minT) / samples;
            incidentLight -= 0.75 * cloudIntensity * cloudShadow * exp(-0.5 * totalIntensity) * (maxT - minT) / samples;
            vec4 fog = computeFogColor(cloudT, ro, rd, lightDir);
            float weight = cloudIntensity * (maxT - minT) / samples * exp(-0.5 * totalIntensity);
            integratedColor += mix(vec3(1.0), fog.rgb, fog.a) * weight;
            integratedWeight += weight;
        }
        if (integratedWeight > 0.0) {
            integratedColor /= integratedWeight;
        } else {
            integratedColor = vec3(1.0);
        }
        float sunAmount = max( dot( rd, lightDir ), 0.0 );
        vec3 fogColor =  mix( vec3(0.5,0.6,0.7), // bluish
        vec3(1.0,0.9,0.7), // yellowish
        clamp(pow(sunAmount,8.0), 0.0, 1.0) );
        incidentLight = clamp(incidentLight, 0.0, 1.0);
        float interpWeight = clamp(1.0 - exp(-totalIntensity * 0.5), 0.0, 1.0);
        if (justIntensity) {
            return vec3(interpWeight);
        }
        color = mix(color, integratedColor * incidentLight, interpWeight);
        return color;

    }
    float integrateClouds(float t, vec3 ro, vec3 rd) {
        float cloudPlaneHeight = 25.0;
        float cloudPlaneRange = 10.0;
        float minT = 0.0;
        float maxT = t;

        float bottomT = (ro.y - (cloudPlaneHeight - cloudPlaneRange)) / -rd.y;
        float topT = (ro.y - (cloudPlaneHeight + cloudPlaneRange)) / -rd.y;

        float enterT = min(bottomT, topT);
        float exitT = max(bottomT, topT);

        if (enterT > minT) {
            minT = enterT;
        }
        if (exitT < maxT) {
            maxT = exitT;
        }
        if (maxT > 1000.0) {
            maxT = 1000.0;
        }
        float totalIntensity = 0.0;
        float samples = 32.0;
        for(float i = 0.0; i < samples; i++) {
            float cloudT = mix(minT, maxT, (i + rand(gl_FragCoord.xy)) / samples);
            vec3 cloudPos = ro + rd * cloudT;
            float cloudIntensity = getCloudDensity(cloudPos, cloudPlaneHeight, cloudPlaneRange);
            totalIntensity += cloudIntensity * (maxT - minT) / samples;
        }
        float interpWeight = clamp(1.0 - exp(-totalIntensity * 0.5), 0.0, 1.0);
        return interpWeight;
    }
    //vec3 computeFogColor( )
    vec3 applyFog(vec3 color, float t, vec3 ro, vec3 rd, vec3 lightDir, bool justFog) {
        vec3 fd = rd;
        if (fd.y == 0.0) fd.y = 0.0001;
        float a = 0.1;
        float b = 0.15;
        float sunAmount = max( dot( rd, lightDir ), 0.0 );
        vec3 fogColor =  mix( vec3(0.5,0.6,0.7), // bluish
        vec3(1.0,0.9,0.7), // yellowish
        clamp(pow(sunAmount,8.0), 0.0, 1.0) );
        if (justFog) {
            fogColor = applyClouds(fogColor, ro, rd, t, lightDir, false);
            return fogColor;
        }
 
        float fogAmount = clamp((a/b) * exp(-ro.y*b) * (1.0-exp( -t*fd.y*b ))/fd.y, 0.0, 1.0);
        color = mix( color, fogColor, fogAmount );
        color = applyClouds(color, ro, rd, t, lightDir, false);
        return color;
    }
    vec3 envMap(vec3 ro, vec3 rd, vec3 lightDir) {
        vec3 col = applyFog(
                vec3(1.0),
                100000.0,
                ro,
                rd,
                lightDir,
                true
            );
        return col;

    }
        
   void main() {
       vec3 lightDir = normalize(vec3(2.0, 1.0, 2.0));
       vec3 viewDir = rayDirection(75.0, vec2(${canvas.width}, ${canvas.height}), gl_FragCoord.xy);
       float heightMagnitude = 25.0;
       vec3 eye = cameraPos;
       vec3 target = eye + cameraDir;
       vec3 up = vec3(0.0, 1.0, 0.0);
       vec3 ro = eye;
       vec3 rd = (makeViewMatrix(eye, target, up) * vec4(viewDir, 1.0)).xyz;
       if (rd.y == 0.0) rd.y = 0.0001;
        vec3 samplePos = ro;


        float t = traceHeightMap(ro, rd, heightMagnitude, 1.0);

        float waterIntersection = (ro.y - 10.0) / -rd.y;
        bool water = false;
        vec3 waterNormal;
        float depthT;
        vec3 depthPos;
        vec3 depthRo;
        vec3 depthRd;
        if (waterIntersection > 0.0 && waterIntersection < t) {
            depthT = t;
            depthPos = ro + rd * depthT;
            depthRo = ro;
            depthRd = rd;
            ro += rd * waterIntersection;
            float normalMagnitude = 0.025;
            vec3 normal = normalize(vec3(normalMagnitude * (
                texture(
                    heightTexture,
                    0.25 * ro.xz - 0.05 * time
                ).r * 2.0 - 1.0
            ), 1.0, normalMagnitude *  (
                texture(
                    heightTexture,
                    0.25 * ro.zx + 0.05 * time
                ).r * 2.0 - 1.0
            )));
            waterNormal = normal;
            rd = reflect(rd,normal);
            t = traceHeightMap(ro, rd, heightMagnitude, 0.1);
            water = true;
        }
        vec3 finalColor = vec3(0.0);
        if (t > 0.0) {
            samplePos = ro + rd * t;
            vec2 finalUv = (samplePos.xz * 0.5 + 0.5) * 0.01  +vec2(0.5);
           // finalUv = floor(finalUv * 2048.0) / 2048.0;
            highp float px = textureLod(heightTexture, finalUv + vec2(1.0 / 2048.0, 0.0), 0.0).r;
            highp float nx = textureLod(heightTexture, finalUv - vec2(1.0 / 2048.0, 0.0), 0.0).r;
            highp float pz = textureLod(heightTexture, finalUv + vec2(0.0, 1.0 / 2048.0), 0.0).r;
            highp float nz = textureLod(heightTexture, finalUv - vec2(0.0, 1.0 / 2048.0), 0.0).r;
            if (px == nx && pz == nz) {
                px = textureLod(heightTexture, finalUv + vec2(1.0 / 1024.0, 0.0), 1.0).r;
                nx = textureLod(heightTexture, finalUv - vec2(1.0 / 1024.0, 0.0), 1.0).r;
                pz = textureLod(heightTexture, finalUv + vec2(0.0, 1.0 / 1024.0), 1.0).r;
                nz = textureLod(heightTexture, finalUv - vec2(0.0, 1.0/ 1024.0), 1.0).r;
            }
            if (px == nx && pz == nz) {
                px = textureLod(heightTexture, finalUv + vec2(1.0 / 512.0, 0.0), 2.0).r;
                nx = textureLod(heightTexture, finalUv - vec2(1.0 / 512.0, 0.0), 2.0).r;
                pz = textureLod(heightTexture, finalUv + vec2(0.0, 1.0 / 512.0), 2.0).r;
                nz = textureLod(heightTexture, finalUv - vec2(0.0, 1.0/ 512.0), 2.0).r;
            }
            vec3 normal = normalize(
                vec3(
                    -heightMagnitude * (px - nx),
                    2.0 * 0.0005,
                   - heightMagnitude * (pz - nz)
                )
            );
            px = textureLod(heightTexture, finalUv + vec2(1.0 / 256.0, 0.0), 3.0).r;
            nx = textureLod(heightTexture, finalUv - vec2(1.0 / 256.0, 0.0), 3.0).r;
            pz = textureLod(heightTexture, finalUv + vec2(0.0, 1.0 / 256.0), 3.0).r;
            nz = textureLod(heightTexture, finalUv - vec2(0.0, 1.0 / 256.0), 3.0).r;
            vec3 lowPassNormal = normalize(
                vec3(
                    -heightMagnitude * (px - nx),
                    2.0 * 0.0005,
                     - heightMagnitude * (pz - nz)
                )
            );
    
            float height = samplePos.y / heightMagnitude;
            float xSample = texture(heightTexture, 0.1 * samplePos.yz).r * 2.0 - 1.0;
            float ySample = texture(heightTexture, 0.1 * samplePos.zx).r * 2.0 - 1.0;
            float zSample = texture(heightTexture, 0.1 * samplePos.xy).r * 2.0 - 1.0;

            vec3 weights = abs(normal);
            weights = weights / (weights.x + weights.y + weights.z);

            float fbmJitter = 0.1 * (weights.x * xSample + weights.y * ySample + weights.z * zSample);
            vec3 color = mix(vec3(0.0, 0.75, 0.0),vec3(0.85, 0.5, 0.0), /*smoothstep(
                0.4,
                0.5,
                height + fbmJitter
            )*/smoothstep(0.0, 1.0, clamp(1.0 - pow(lowPassNormal.y, 0.15) + fbmJitter, 0.0, 1.0)));

            color = mix(vec3(0.95, 0.85, 0.7), color, smoothstep(
                0.4,
                0.5,
                height + fbmJitter
            ));
            color = mix(color, vec3(0.9), smoothstep(
                0.7,
                0.75,
                height + fbmJitter
            ));
            color *= (1.0 + 2.0 * fbmJitter);
            
            float dir = max(dot(normal, lightDir), 0.0);
            vec3 shadowLightDir = lightDir;
            float shadow = traceShadowRay(samplePos + lightDir * 0.02, shadowLightDir, heightMagnitude, 0.01);

            dir *= shadow;         
            float cloudShadow = 1.0 - integrateClouds(
                50.0,
                samplePos,
                vec3(0.0, 1.0, 0.0)
            );
            float ambient = 0.3;
            float finalAo = 1.0;
            for(float i = 4.0; i <= 8.0; i++) {
                finalAo *= clamp(1.0 - 10.0 / pow(2.0, i - 4.0) * max( textureLod(heightTexture, finalUv, i).r - height, 0.0), 0.0, 1.0);
            }
            vec3 groundColor = vec3(0.0, 0.0, 0.5);
            vec3 brownColor = vec3(0.425, 0.25, 0.25);
            vec3 skyColor = vec3(0.0, 0.7, 1.0);
            vec3 bounce = vec3(0.0);
            if (normal.y < 0.0) {
                bounce = mix(
                    groundColor,
                    brownColor,
                    1.0 + normal.y
                );
            } else {
                bounce = mix(
                    brownColor,
                    skyColor,
                    normal.y
                );
            }
            finalColor = (color * ambient * sqrt(finalAo) + color * dir * finalAo + (bounce * color * finalAo) / 3.14159) * (0.25 + 0.75 * cloudShadow);
            finalColor = applyFog(finalColor, t, ro, rd, lightDir, false);


        } else {
            finalColor = envMap(ro, rd, lightDir);
        }
        if (water) {
            finalColor = mix(finalColor, vec3(0.0, exp(-0.5 * clamp(distance(depthPos, ro),0.0, 1.0)), 1.0),0.0* pow(
                max(dot(-depthRd, waterNormal), 0.0), 1.0
            ));
            finalColor *= max(dot(waterNormal, lightDir), 0.0);
            if (t <= 0.0) {
                float specular = pow(max(dot(rd, lightDir), 0.0), 64.0);
                finalColor += vec3(1.0) * specular;
            }
            float weight = (1.0 / (pow(1.0 + (10.0 - depthPos.y), 15.0))) * min(20.0 * (10.0 - depthPos.y), 1.0);
            if (weight > 1.0 / 255.0) {
            finalColor = mix(finalColor, vec3(0.75), (0.5 + 0.5 * sin((20.0 + 0.1 * fbm(vec3(0.01 * ro.xz, 0.001 * time))) * depthPos.y - 5.0 * time + 20.0 * fbm(vec3(0.01 * ro.xz, 0.001 * time))))*weight);
            }

            finalColor = applyFog(finalColor, waterIntersection, depthRo, depthRd, lightDir, false);

        }
        pc_FragColor = vec4(finalColor, 1.0);
   }
   `, // Paste your original fragment shader source here
     fragmentColorMap: /*glsl*/ `#version 300 es
     precision highp float;
     layout(location = 0) out vec4 pc_FragColor;
        in vec2 vPosition;
        void main() {
            // Dummy shader that makes a radial gradient.
            float dist = length(vPosition);
            vec3 color = vec3(1.0-dist, 0.0, dist);
            pc_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
    `,
     fragmentHeightMap: /*glsl*/ `#version 300 es
     precision highp float;
     layout(location = 0) out vec4 pc_FragColor;
        in vec2 vPosition;
        // Simplex 2D noise
        vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

        float snoise(vec2 v){
        const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                -0.577350269189626, 0.024390243902439);
        vec2 i  = floor(v + dot(v, C.yy) );
        vec2 x0 = v -   i + dot(i, C.xx);
        vec2 i1;
        i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz;
        x12.xy -= i1;
        i = mod(i, 289.0);
        vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));
        vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
            dot(x12.zw,x12.zw)), 0.0);
        m = m*m ;
        m = m*m ;
        vec3 x = 2.0 * fract(p * C.www) - 1.0;
        vec3 h = abs(x) - 0.5;
        vec3 ox = floor(x + 0.5);
        vec3 a0 = x - ox;
        m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
        vec3 g;
        g.x  = a0.x  * x0.x  + h.x  * x0.y;
        g.yz = a0.yz * x12.xz + h.yz * x12.yw;
        return 130.0 * dot(m, g);
        }
        #define NUM_OCTAVES 8

        float fbm(vec2 x) {
            float v = 0.0;
            float a = 0.5;
            vec2 shift = vec2(100);
            // Rotate to reduce axial bias
            mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
            for (int i = 0; i < NUM_OCTAVES; ++i) {
                v += a * snoise(x);
                x = rot * x * 2.0 + shift;
                a *= 0.5;
            }
            return v;
        }
        float blendFactor(float pos) {
            const float blendRange = 0.1; // adjust this value to change the size of the blend area
            return smoothstep(0.0, blendRange, pos) * smoothstep(1.0, 1.0 - blendRange, pos);
        }
        void main() {
            // Dummy shader that generates a height map.
            // The y coordinate is used as the height value.
            float noiseStart = fbm(vPosition * 2.5);
            float noiseXMirror = fbm(vec2(-vPosition.x, vPosition.y) * 2.5);
            float noiseYMirror = fbm(vec2(vPosition.x, -vPosition.y) * 2.5);
            float noiseXYMirror = fbm(vec2(-vPosition.x, -vPosition.y) * 2.5);

            // Blend at edges 
            float blendX = 0.5 * pow(abs(vPosition.x), 10.0);
            float blendY = 0.5 * pow(abs(vPosition.y), 10.0);

            float noiseXBlended = mix(noiseStart, noiseXMirror, blendX);
            float noiseXYBlended = mix(noiseYMirror, noiseXYMirror, blendX);
            float noiseBlended = mix(noiseXBlended, noiseXYBlended, blendY);
                       
            pc_FragColor = vec4(vec3(
                0.5 + 0.5 * noiseBlended
            ), 1.0);
        }
    `,
 };

 var shaders = {
     vertex: compileShader(gl, gl.VERTEX_SHADER, shaderSources.vertex),
     fragmentMain: compileShader(gl, gl.FRAGMENT_SHADER, shaderSources.fragmentMain),
     fragmentColorMap: compileShader(gl, gl.FRAGMENT_SHADER, shaderSources.fragmentColorMap),
     fragmentHeightMap: compileShader(gl, gl.FRAGMENT_SHADER, shaderSources.fragmentHeightMap),
 };

 // Create the programs
 var programs = {
     main: createProgram(gl, shaders.vertex, shaders.fragmentMain),
     colorMap: createProgram(gl, shaders.vertex, shaders.fragmentColorMap),
     heightMap: createProgram(gl, shaders.vertex, shaders.fragmentHeightMap),
 };

 var textures = {
         colorMap: [2048, 2048],
         heightMap: [2048, 2048],
     }
     // Create the buffer
 var positionBuffer = createBuffer(gl, [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, ]);

 // Create the textures
 var colorTexture = createTexture(gl, gl.RGBA, 2048, 2048, gl.UNSIGNED_BYTE, gl.RGBA8);
 var heightTexture = createTexture(gl, gl.RED, 2048, 2048, gl.UNSIGNED_BYTE, gl.R8);

 // Create the framebuffers
 var colorFBO = createFramebuffer(gl, colorTexture);
 var heightFBO = createFramebuffer(gl, heightTexture);

 // Clear the canvas and render
 gl.clearColor(0, 0, 0, 0);

 function checkTimerQuery(timerQuery, gl) {
     const available = gl.getQueryParameter(timerQuery, gl.QUERY_RESULT_AVAILABLE);
     if (available) {
         const elapsedTimeInNs = gl.getQueryParameter(timerQuery, gl.QUERY_RESULT);
         const elapsedTimeInMs = elapsedTimeInNs / 1000000;
         console.log(elapsedTimeInMs);
     } else {
         // If the result is not available yet, check again after a delay
         setTimeout(() => {
             checkTimerQuery(timerQuery, gl);
         }, 1);
     }
 }
 renderToTexture(gl, programs.colorMap, colorFBO, positionBuffer, [2048, 2048]);
 renderToTexture(gl, programs.heightMap, heightFBO, positionBuffer, [2048, 2048]);
 gl.bindTexture(gl.TEXTURE_2D, heightTexture);
 gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
 gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
 gl.generateMipmap(gl.TEXTURE_2D);
 const stats = new Stats();
 stats.showPanel(0);
 document.body.appendChild(stats.dom);
 const keys = {};
 document.addEventListener('keydown', (e) => {
     keys[e.key] = true;
 });
 document.addEventListener('keyup', (e) => {
     keys[e.key] = false;
 });
 document.addEventListener('click', e => {
     canvas.requestPointerLock();
 })
 document.addEventListener('mousemove', e => {
     if (document.pointerLockElement === canvas) {
         xRotation += e.movementX * 0.001;
         yRotation -= e.movementY * 0.001;
         yRotation = Math.max(Math.min(yRotation, Math.PI / 2), -Math.PI / 2);
     }
 })
 let cameraPos = [0, 20, 0];
 let xRotation = 0;
 let yRotation = 0;

 function render() {
     let cameraDir = [0, 0, 0]
     cameraDir[0] = Math.cos(xRotation) * Math.cos(yRotation);
     cameraDir[1] = Math.sin(yRotation);
     cameraDir[2] = Math.sin(xRotation) * Math.cos(yRotation);
     const forward = cameraDir;
     // We don't have a library, so we have to do the math ourselves
     const right = [
         Math.cos(xRotation + Math.PI / 2) * Math.cos(yRotation),
         Math.sin(yRotation),
         Math.sin(xRotation + Math.PI / 2) * Math.cos(yRotation)
     ];
     if (keys['w']) {
         cameraPos[0] += forward[0];
         cameraPos[1] += forward[1];
         cameraPos[2] += forward[2];
     }
     if (keys['s']) {
         cameraPos[0] -= forward[0];
         cameraPos[1] -= forward[1];
         cameraPos[2] -= forward[2];
     }
     if (keys['a']) {
         cameraPos[0] -= right[0];
         cameraPos[1] -= right[1];
         cameraPos[2] -= right[2];
     }
     if (keys['d']) {
         cameraPos[0] += right[0];
         cameraPos[1] += right[1];
         cameraPos[2] += right[2];
     }

     stats.begin();
     // Pass the color and height textures to the main program
     gl.useProgram(programs.main);
     var colorTextureLocation = gl.getUniformLocation(programs.main, 'colorTexture');
     var heightTextureLocation = gl.getUniformLocation(programs.main, 'heightTexture');
     var timeLocation = gl.getUniformLocation(programs.main, 'time');
     var posLocation = gl.getUniformLocation(programs.main, 'cameraPos');
     var dirLocation = gl.getUniformLocation(programs.main, 'cameraDir');
     gl.uniform1i(colorTextureLocation, 0); // texture unit 0
     gl.uniform1i(heightTextureLocation, 1); // texture unit 1

     gl.uniform3fv(posLocation, cameraPos);

     gl.uniform3fv(dirLocation, cameraDir);

     gl.activeTexture(gl.TEXTURE0);
     gl.bindTexture(gl.TEXTURE_2D, colorTexture);
     gl.activeTexture(gl.TEXTURE1);
     gl.bindTexture(gl.TEXTURE_2D, heightTexture);

     gl.uniform1f(timeLocation, performance.now() / 1000);


     const queryExt = gl.getExtension('EXT_disjoint_timer_query_webgl2');
     const timerQuery = gl.createQuery();
     gl.beginQuery(queryExt.TIME_ELAPSED_EXT, timerQuery);
     renderToScreen(gl, programs.main, positionBuffer);
     gl.endQuery(queryExt.TIME_ELAPSED_EXT);
     checkTimerQuery(timerQuery, gl);
     stats.end();

     // Loop at 60fps
     requestAnimationFrame(render);
 }

 // Start rendering
 render();

 // Helper functions
 function compileShader(gl, type, source) {
     var shader = gl.createShader(type);
     gl.shaderSource(shader, source);
     gl.compileShader(shader);
     if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
         console.error('Shader failed to compile: ' + gl.getShaderInfoLog(shader));
     }
     return shader;
 }

 function createProgram(gl, vertexShader, fragmentShader) {
     var program = gl.createProgram();
     gl.attachShader(program, vertexShader);
     gl.attachShader(program, fragmentShader);
     gl.linkProgram(program);
     if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
         console.error('Program failed to link: ' + gl.getProgramInfoLog(program));
     }
     return program;
 }

 function createBuffer(gl, data) {
     var buffer = gl.createBuffer();
     gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
     gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
     return buffer;
 }

 function createTexture(gl, format, width, height, type, internalFormat = format) {
     var texture = gl.createTexture();
     gl.bindTexture(gl.TEXTURE_2D, texture);
     if (gl.getError() != gl.NO_ERROR) {
         console.error("Error binding texture");
     }

     gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
     if (gl.getError() != gl.NO_ERROR) {
         console.error("Error setting texture image");
     }
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

     // Repeat
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

     return texture;
 }

 function createFramebuffer(gl, texture) {
     var fbo = gl.createFramebuffer();
     gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
     gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
     gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
     return fbo;
 }

 function renderToTexture(gl, program, fbo, positionBuffer, size) {
     gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
     gl.viewport(0, 0, size[0], size[1]);
     renderWithProgram(gl, program, positionBuffer);
 }

 function renderToScreen(gl, program, positionBuffer) {
     gl.bindFramebuffer(gl.FRAMEBUFFER, null);
     gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
     renderWithProgram(gl, program, positionBuffer);
 }

 function renderWithProgram(gl, program, positionBuffer) {
     gl.useProgram(program);
     gl.enableVertexAttribArray(0);
     gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
     gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
     gl.drawArrays(gl.TRIANGLES, 0, 6);
 }