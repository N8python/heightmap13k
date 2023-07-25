#version 300 es
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