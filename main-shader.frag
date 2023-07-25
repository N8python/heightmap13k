#version 300 es
     precision highp float;
     layout(location = 0) out vec4 pc_FragColor;
    uniform sampler2D heightTexture;
    uniform vec3 cameraPos;
    uniform vec3 cameraDir;
    uniform vec2 resolution;
    uniform highp float time;
   in vec2 vPosition;
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

                dt = max(0.4 * abs(p.y - h), 0.001);
                lh = h;
                ly = p.y;
                t += dt;
            }
           
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
        vec3 fogColor =  mix( vec3(0.5,0.6,0.7),
        vec3(1.0,0.9,0.7),
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

    vec3 applyFog(vec3 color, float t, vec3 ro, vec3 rd, vec3 lightDir, bool justFog) {
        vec3 fd = rd;
        if (fd.y == 0.0) fd.y = 0.0001;
        float a = 0.1;
        float b = 0.15;
        float sunAmount = max( dot( rd, lightDir ), 0.0 );
        vec3 fogColor =  mix( vec3(0.5,0.6,0.7),
        vec3(1.0,0.9,0.7), 
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
    highp float fetchTexture(sampler2D texture, vec2 uv, vec2 offset, float lod) {
        return textureLod(texture, uv + offset, lod).r;
    }
    
    vec3 computeNormal(float px, float nx, float pz, float nz, float heightMagnitude) {
        return normalize(
            vec3(
                -heightMagnitude * (px - nx),
                2.0 * 0.0005,
                -heightMagnitude * (pz - nz)
            )
        );
    }
    vec4 fetchTextureVec4(sampler2D texture, vec2 uv, float div, float lod) {
        float px = fetchTexture(texture, uv, vec2(1.0 / div, 0.0), lod);
        float nx = fetchTexture(texture, uv, vec2(-1.0 / div, 0.0), lod);
        float pz = fetchTexture(texture, uv, vec2(0.0, 1.0 / div), lod);
        float nz = fetchTexture(texture, uv, vec2(0.0, -1.0 / div), lod);
        return vec4(px, nx, pz, nz);
    }
   void main() {
       vec3 lightDir = normalize(vec3(2.0, 1.0, 2.0));
       vec3 viewDir = rayDirection(75.0, resolution, gl_FragCoord.xy);
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
            float lod = 0.0;
            float div = 2048.0;
            vec4 textures;
            
            for(int i = 0; i < 3; i++) {
                textures = fetchTextureVec4(heightTexture, finalUv, div, lod);
                if (textures.x != textures.y || textures.z != textures.w) {
                    break;
                }
                lod += 1.0;
                div /= 2.0;
            }
            vec3 normal = computeNormal(textures.x, textures.y, textures.z, textures.w, heightMagnitude);
            textures = fetchTextureVec4(heightTexture, finalUv, 256.0, 3.0);
            vec3 lowPassNormal = computeNormal(textures.x, textures.y, textures.z, textures.w, heightMagnitude);
            float height = samplePos.y / heightMagnitude;
            float xSample = texture(heightTexture, 0.1 * samplePos.yz).r * 2.0 - 1.0;
            float ySample = texture(heightTexture, 0.1 * samplePos.zx).r * 2.0 - 1.0;
            float zSample = texture(heightTexture, 0.1 * samplePos.xy).r * 2.0 - 1.0;

            vec3 weights = abs(normal);
            weights = weights / (weights.x + weights.y + weights.z);

            float fbmJitter = 0.1 * (weights.x * xSample + weights.y * ySample + weights.z * zSample);
            vec3 color = mix(vec3(0.0, 0.75, 0.0),vec3(0.85, 0.5, 0.0), smoothstep(0.0, 1.0, clamp(1.0 - pow(lowPassNormal.y, 0.15) + fbmJitter, 0.0, 1.0)));

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
            finalColor = mix(finalColor, 0.8 * vec3(0.0, 0.9 * exp(-0.35 * clamp(distance(depthPos, ro),0.0, 1.0)), 1.0), max(dot(-depthRd, vec3(0.0, 1.0, 0.0)), 0.0));
            finalColor *= 0.75 + 0.25 * max(dot(waterNormal, lightDir), 0.0);
            if (t <= 0.0) {
                float specular = pow(max(dot(rd, lightDir), 0.0), 64.0);
                finalColor += vec3(1.0) * specular;
            }
            float weight = (1.0 / (pow(1.0 + (10.0 - depthPos.y), 15.0))) * min(20.0 * (10.0 - depthPos.y), 1.0);
            if (weight > 1.0 / 255.0) {
            finalColor = mix(finalColor, vec3(0.75), (0.5 + 0.5 * sin((20.0 + 0.1 * (2.0 * textureLod(heightTexture, 0.01 * ro.xz - 0.001 * time, 0.0).r - 1.0)) * depthPos.y - 5.0 * time + 20.0 * (2.0 * textureLod(heightTexture, 0.01 * ro.zx + 0.001 * time, 0.0).r - 1.0)))*weight);
            }

            finalColor = applyFog(finalColor, waterIntersection, depthRo, depthRd, lightDir, false);

        }
        pc_FragColor = vec4(finalColor, 1.0);
   }