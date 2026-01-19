#define S(a,b,t) smoothstep(a,b,t)
#define NUM_LAYERS 4.

// -------------------------------
// Noise & line functions
float N21(vec2 p){
    vec3 a = fract(vec3(p.xyx)*vec3(213.897,653.453,253.098));
    a += dot(a,a.yzx+79.76);
    return fract((a.x+a.y)*a.z);
}

vec2 GetPos(vec2 id, vec2 offs, float t){
    float n = N21(id+offs);
    float n1 = fract(n*10.0);
    float n2 = fract(n*100.0);
    float a = t+n;
    return offs + vec2(sin(a*n1),cos(a*n2))*0.4;
}

float df_line(vec2 a, vec2 b, vec2 p){
    vec2 pa = p-a;
    vec2 ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa - ba*h);
}

float line(vec2 a, vec2 b, vec2 uv){
    float r1 = 0.04;
    float r2 = 0.01;
    float d = df_line(a,b,uv);
    float d2 = length(a-b);
    float fade = S(1.5,0.5,d2);
    fade += S(0.05,0.02,abs(d2-0.75));
    return S(r1,r2,d)*fade;
}

float NetLayer(vec2 st, float n, float t){
    vec2 id = floor(st)+n;
    st = fract(st)-0.5;
    vec2 p[9];
    int i=0;
    for(float y=-1.0;y<=1.0;y++)
        for(float x=-1.0;x<=1.0;x++)
            p[i++] = GetPos(id,vec2(x,y),t);
    float m=0.0;
    float sparkle=0.0;
    for(int i=0;i<9;i++){
        m += line(p[4],p[i],st);
        float d = length(st-p[i]);
        float s = 0.005/(d*d);
        s *= S(1.0,0.7,d);
        float pulse = sin((fract(p[i].x)+fract(p[i].y)+t)*5.0)*0.4+0.6;
        pulse = pow(pulse,20.0);
        s *= pulse;
        sparkle += s;
    }
    m += line(p[1],p[3],st);
    m += line(p[1],p[5],st);
    m += line(p[7],p[5],st);
    m += line(p[7],p[3],st);
    float sPhase = (sin(t+n)+sin(t*0.1))*0.25 + 0.5;
    sPhase += pow(sin(t*0.1)*0.5+0.5,50.0)*5.0;
    m += sparkle*sPhase;
    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = (fragCoord-iResolution.xy*0.5)/iResolution.y;

    // -------------------------------
    // Use pre-computed envelope uniforms (proper AR envelope from Python)
    // Available uniforms:
    //   envelope  - master AR envelope (5ms attack, 150ms release)
    //   bass_env  - bass-only envelope (10ms attack, 200ms release)
    //   transient - fast response (2ms/50ms) - catches hits
    //   sustain   - slow response (50ms/300ms) - held energy
    // -------------------------------
    float envZoom   = envelope;           // zoom responds to master envelope
    float envSpeed  = envelope * 0.7;     // speed modulation
    float envBright = transient * 0.9;    // brightness uses transient for punch

    // -------------------------------
    // Time modulation
    // -------------------------------
    float tBase = audio_time*0.15;
    float t = mix(tBase, tBase*1.6, envSpeed);

    // -------------------------------
    // Global zoom
    // -------------------------------
    float zoom = 1.0 + envZoom;
    vec2 st = uv*zoom;

    // -------------------------------
    // Rotation
    // -------------------------------
    float baseRot = audio_time*0.1;
    float kickRot = envZoom*0.05;
    float ang = baseRot + kickRot;
    float s = sin(ang);
    float c = cos(ang);
    mat2 rot = mat2(c,-s,s,c);
    st *= rot;

    // -------------------------------
    // Render layers
    // -------------------------------
    float m = 0.0;
    for(float i=0.0;i<1.0;i+=1.0/NUM_LAYERS){
        float z = fract(t+i+7.3);
        float fade = S(0.0,0.6,z)*S(1.0,0.8,z);
        float size = mix(15.0,1.0,z)*(1.0+envZoom*0.8);
        m += fade*NetLayer(st*size,i,t);
    }

    // -------------------------------
    // Color
    // -------------------------------
    vec3 baseCol = vec3(
        sin(baseRot),
        cos(baseRot*0.4),
        -sin(baseRot*0.24)
    )*0.4 + 0.6;

    vec3 col = baseCol*m;
    col *= 1.0 + envBright*0.25;
    col *= 1.0 - dot(uv,uv);
    col *= S(0.0,3.0,iTime);

    fragColor = vec4(col,1.0);
}
