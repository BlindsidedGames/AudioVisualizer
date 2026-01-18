// "Neural Web" - Audio Reactive Network
// Based on "The Universe Within" by BigWIngs (Shadertoy lscczl)
// Modified for smooth audio reactivity without time jerking
//
// Audio uniforms: bass, mids, highs, volume, beat (all 0.0 - 1.0)
// iTime: constant forward time, audio_time: audio-energy accumulated time
//
// Effects:
// - Smooth zoom pulse on beats (no backwards motion)
// - Line brightness responds to volume
// - Sparkle intensity tied to highs
// - Color shifts with bass/mids/highs
// - Rotation speed slightly modulated by energy

#define S(a, b, t) smoothstep(a, b, t)
#define NUM_LAYERS 3.

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

float NetLayer(vec2 st, float n, float t, float lineThickness, float sparkleBoost) {
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

    // Sparkle phase with audio boost
    float sPhase = (sin(t + n) + sin(t * .1)) * .25 + .5;
    sPhase += pow(sin(t * .1) * .5 + .5, 50.) * 5.;
    sPhase *= sparkleBoost;

    m += sparkle * sPhase;

    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;

    // === TIME - NEVER GOES BACKWARDS ===
    // Use iTime for consistent forward motion
    // Add offset so layers start spread out (not all at same phase = grid look)
    float t = iTime * 0.07 + 50.0;  // Slower speed + offset to start "initialized"

    // === ROTATION - SMOOTH AND CONSISTENT ===
    // Base rotation always moves forward, slight speed variation from volume
    float rotSpeed = 0.03 + volume * 0.02;  // Slower rotation
    float rotT = iTime * rotSpeed;

    float s = sin(rotT);
    float c = cos(rotT);
    mat2 rot = mat2(c, -s, s, c);
    vec2 st = uv * rot;

    // === ZOOM PULSE - ADDITIVE ONLY ===
    // Beat causes zoom IN (scale up), never backwards
    // Use smoothed values to prevent jerking
    float beatSmooth = beat * beat;  // Square it for snappier but smooth response
    float zoomPulse = 1.0 + beatSmooth * 0.15 + bass * 0.1;
    st *= zoomPulse;

    // === LINE THICKNESS - RESPONDS TO BASS ===
    float lineThickness = 1.0 + bass * 0.8;

    // === SPARKLE BOOST - RESPONDS TO HIGHS ===
    float sparkleBoost = 1.0 + highs * 1.5 + beat * 0.5;

    // === RENDER LAYERS ===
    float m = 0.;
    for (float i = 0.; i < 1.; i += 1. / NUM_LAYERS) {
        float z = fract(t + i);
        float size = mix(25., 1., z);
        float fade = S(0., .6, z) * S(1., .8, z);

        m += fade * NetLayer(st * size, i, iTime, lineThickness, sparkleBoost);
    }

    // === BASE COLORS - CYCLE SMOOTHLY ===
    vec3 baseCol = vec3(
        sin(rotT),
        cos(rotT * .4),
        -sin(rotT * .24)
    ) * .4 + .6;

    // === COLOR TINTING FROM AUDIO ===
    baseCol.r += bass * 0.3;      // Red pulses with bass
    baseCol.g += mids * 0.2;      // Green with mids
    baseCol.b += highs * 0.35;    // Blue with highs

    vec3 col = baseCol * m;

    // === BRIGHTNESS PULSE ON BEAT ===
    // Additive brightness, never subtracts
    col *= 1.0 + beat * 0.5 + volume * 0.2;

    // === GLOW FROM BASS ===
    float glow = max(0.0, -uv.y) * bass * 0.5;
    col += baseCol * glow;

    // === VIGNETTE ===
    col *= 1. - dot(uv, uv) * 0.8;

    // === FADE IN/OUT (for long renders) ===
    float fadeTime = mod(iTime, 230.);
    col *= S(0., 5., fadeTime) * S(230., 220., fadeTime);

    fragColor = vec4(col, 1);
}
