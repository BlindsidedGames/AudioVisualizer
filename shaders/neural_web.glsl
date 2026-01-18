// "Neural Web" - Audio Reactive Network
// Based on "The Universe Within" by BigWIngs (Shadertoy lscczl)
//
// Audio reactivity is smooth - no sudden jumps
// Audio uniforms: bass, mids, highs, volume, beat (all 0.0 - 1.0, pre-smoothed)

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

float line(vec2 a, vec2 b, vec2 uv) {
    float r1 = .04;
    float r2 = .01;

    float d = df_line(a, b, uv);
    float d2 = length(a - b);
    float fade = S(1.5, .5, d2);

    fade += S(.05, .02, abs(d2 - .75));
    return S(r1, r2, d) * fade;
}

float NetLayer(vec2 st, float n, float t) {
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
        m += line(p[4], p[i], st);

        float d = length(st - p[i]);
        float s = (.005 / (d * d));
        s *= S(1., .7, d);

        float pulse = sin((fract(p[i].x) + fract(p[i].y) + t) * 5.) * .4 + .6;
        pulse = pow(pulse, 20.);

        s *= pulse;
        sparkle += s;
    }

    m += line(p[1], p[3], st);
    m += line(p[1], p[5], st);
    m += line(p[7], p[5], st);
    m += line(p[7], p[3], st);

    float sPhase = (sin(t + n) + sin(t * .1)) * .25 + .5;
    sPhase += pow(sin(t * .1) * .5 + .5, 50.) * 5.;

    m += sparkle * sPhase;

    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;

    // Use audio_time for smooth audio-reactive animation speed
    // audio_time accumulates smoothly - never jumps backwards
    float t = audio_time * 0.15;

    // Rotation based on audio_time - smooth, never jerky
    float rotT = audio_time * 0.08;

    float s = sin(rotT);
    float c = cos(rotT);
    mat2 rot = mat2(c, -s, s, c);
    vec2 st = uv * rot;

    // Render layers - offset so they don't start aligned
    float m = 0.;
    for (float i = 0.; i < 1.; i += 1. / NUM_LAYERS) {
        float z = fract(t + i + 7.3);
        float size = mix(15., 1., z);
        float fade = S(0., .6, z) * S(1., .8, z);

        // Node animation also uses audio_time for consistency
        m += fade * NetLayer(st * size, i, audio_time * 0.5);
    }

    // Base colors cycle with audio_time
    vec3 baseCol = vec3(
        sin(rotT),
        cos(rotT * .4),
        -sin(rotT * .24)
    ) * .4 + .6;

    // Subtle color tinting from audio (these are pre-smoothed now)
    baseCol.r += bass * 0.2;
    baseCol.b += highs * 0.2;

    vec3 col = baseCol * m;

    // Gentle brightness on beat (pre-smoothed)
    col *= 1.0 + beat * 0.25;

    // Vignette
    col *= 1. - dot(uv, uv);

    // Fade in at start
    col *= S(0., 3., iTime);

    fragColor = vec4(col, 1);
}
