// "The Universe Within" - Fully Audio Reactive
// Fixed undefined variables and enhanced beat response

#define S(a, b, t) smoothstep(a, b, t)
#define NUM_LAYERS 4.

float N21(vec2 p) {
    vec3 a = fract(vec3(p.xyx) * vec3(213.897, 653.453, 253.098));
    a += dot(a, a.yzx + 79.76);
    return fract((a.x + a.y) * a.z);
}

vec2 GetPos(vec2 id, vec2 offs, float t) {
    float n = N21(id + offs);
    float n1 = fract(n * 10.);
    float n2 = fract(n * 100.);
    float a = t + n;
    return offs + vec2(sin(a * n1), cos(a * n2)) * .4;
}

float df_line(in vec2 a, in vec2 b, in vec2 p) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
    return length(pa - ba * h);
}

float line(vec2 a, vec2 b, vec2 uv, float thickness) {
    float r1 = .04 * thickness;
    float r2 = .01 * thickness;

    float d = df_line(a, b, uv);
    float d2 = length(a - b);
    float fade = S(1.5, .5, d2);

    fade += S(.05, .02, abs(d2 - .75));
    return S(r1, r2, d) * fade;
}

float NetLayer(vec2 st, float n, float t, float lineThickness, float softBeat) {
    vec2 id = floor(st) + n;
    st = fract(st) - .5;

    vec2 p[9];
    int i = 0;
    for (float y = -1.; y <= 1.; y++) {
        for (float x = -1.; x <= 1.; x++) {
            p[i++] = GetPos(id, vec2(x, y), t);
        }
    }

    float m = 0.;
    float sparkle = 0.;

    for (int i = 0; i < 9; i++) {
        m += line(p[4], p[i], st, lineThickness);

        float d = length(st - p[i]);

        float s = (.005 / (d * d));
        s *= S(1., .7, d);

        float pulse = sin((fract(p[i].x) + fract(p[i].y) + t) * 5.) * .4 + .6;
        pulse = pow(pulse, 20.);

        s *= pulse;
        sparkle += s;
    }

    m += line(p[1], p[3], st, lineThickness);
    m += line(p[1], p[5], st, lineThickness);
    m += line(p[7], p[5], st, lineThickness);
    m += line(p[7], p[3], st, lineThickness);

    // Sparkle phase - enhanced by audio
    float sPhase = (sin(t + n) + sin(t * .1)) * .25 + .5;
    sPhase += pow(sin(t * .1) * .5 + .5, 50.) * 5.;
    sPhase *= (1.0 + softBeat * 0.8); // Sparkle more on beats

    m += sparkle * sPhase;

    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // --- SOFTENED BEAT --- 
    float softBeat = smoothstep(0.2, 0.8, beat); 
    softBeat = pow(softBeat, 0.6); // compress spikes

    vec2 uv = (fragCoord - iResolution.xy * 0.5) / iResolution.y;

    // Animation time from audio energy
    float t = audio_time * 0.75;

    // === BEAT-REACTIVE ROTATION ===
    float rotSpeed = 0.06 + bass * 0.08;
    float rotT = iTime * rotSpeed;

    rotT += softBeat * 0.15;  // subtle beat rotation
    rotT += softBeat * 0.5;   // strong beat rotation kick

    float s = sin(rotT);
    float c = cos(rotT);
    mat2 rot = mat2(c, -s, s, c);
    vec2 st = uv * rot;

    // === BEAT-REACTIVE ZOOM/PULSE ===
    float beatPulse = 1.8 + softBeat * 0.18 + bass * 0.06;
    st *= beatPulse;

    // === BEAT-REACTIVE LINE THICKNESS ===
    float lineThickness = 1.0 + bass * 1.1 + softBeat * 0.5;

    float m = 0.;
    for (float i = 0.; i < 1.; i += 1. / NUM_LAYERS) {
        float z = fract(t + i);

        float size = mix(15., 1., z);
        float fade = S(0., .6, z) * S(1., .8, z);

        m += fade * NetLayer(st * size, i, t, lineThickness, softBeat);
    }

    // Base colors cycle over time
    vec3 baseCol = vec3(
        sin(rotT) + 0.1,
        cos(rotT * 0.4) + 0.1,
        -sin(rotT * 0.24) + 0.2
    ) * 0.4 + 0.6;

    // === AUDIO-RESPONSIVE COLORS ===
    baseCol.r += pow(bass, 0.7) * 0.25;
    baseCol.g += pow(mids, 0.8) * 0.15;
    baseCol.b += pow(highs, 0.7) * 0.25;

    vec3 col = baseCol * m;

    // === BEAT-REACTIVE BRIGHTNESS ===
    col *= 1.0 + softBeat * 0.35 + bass * 0.2 + volume * 0.2;

    // Vignette (slightly reduced on beats)
    float vignette = 1.0 - dot(uv, uv) * 0.75;
    col *= vignette + softBeat * 0.1;

    fragColor = vec4(col, 1.0);
}
