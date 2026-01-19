// "The Universe Within" - Audio Reactive Version (SLOW SPEED)
// Original by BigWIngs on Shadertoy (lscczl)
// Modified for audio reactivity
//
// Audio uniforms: bass, mids, highs, volume, beat (all 0.0 - 1.0)
// Speed: 25% (0.25x multiplier) - Very slow/ambient pace

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

    float lineThickness = 1.0;

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

    // Original sparkle phase - unchanged
    float sPhase = (sin(t + n) + sin(t * .1)) * .25 + .5;
    sPhase += pow(sin(t * .1) * .5 + .5, 50.) * 5.;

    m += sparkle * sPhase;

    return m;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy * .5) / iResolution.y;

    // Apply zoom (zoom uniform from app, default 1.0)
    float z_factor = max(zoom, 0.1);  // Prevent division by zero
    uv *= z_factor;

    // Use pre-computed audio_time which accumulates smoothly based on audio energy
    // Speed multiplier: 0.25x (slow - 25% speed, very ambient)
    float t = audio_time * 0.25;

    // Rotation uses regular time for smooth consistent rotation
    float rotT = iTime * 0.1;
    float s = sin(rotT);
    float c = cos(rotT);
    mat2 rot = mat2(c, -s, s, c);
    vec2 st = uv * rot;

    float m = 0.;
    for (float i = 0.; i < 1.; i += 1. / NUM_LAYERS) {
        float z = fract(t + i);

        float size = mix(15., 1., z);
        float fade = S(0., .6, z) * S(1., .8, z);

        m += fade * NetLayer(st * size, i, t);
    }

    // Base colors cycle over time
    vec3 baseCol = vec3(
        sin(rotT) + 0.1,
        cos(rotT * .4) + 0.1,
        -sin(rotT * .24) + 0.2
    ) * .4 + .6;

    // Subtle color tinting from audio
    baseCol.r += bass * 0.15;
    baseCol.b += highs * 0.15;

    vec3 col = baseCol * m;

    // Vignette - adjusted for zoom
    col *= 1. - dot(uv, uv) / (z_factor * z_factor);

    // Apply fade (for fade-out at end of video)
    col *= fade;

    fragColor = vec4(col, 1);
}
