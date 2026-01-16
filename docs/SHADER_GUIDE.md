# Shader Customization Guide

This guide explains how to create, modify, and add custom shaders to the Audio Visualizer.

## Quick Start

1. Create a new `.glsl` file in the `shaders/` folder
2. Restart the visualizer - your shader will appear in the dropdown
3. That's it!

## Shader Structure

Shaders use the **Shadertoy** format. Here's a minimal template:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalize coordinates (0 to 1)
    vec2 uv = fragCoord / iResolution.xy;

    // Your visualization code here
    vec3 col = vec3(uv.x, uv.y, 0.5);

    // Output color
    fragColor = vec4(col, 1.0);
}
```

## Available Uniforms

These variables are automatically provided to your shader:

### Standard Uniforms
| Uniform | Type | Description |
|---------|------|-------------|
| `iResolution` | `vec2` | Output resolution (width, height) |
| `iTime` | `float` | Time in seconds since start |
| `iTimeDelta` | `float` | Time since last frame |
| `iFrame` | `int` | Current frame number |

### Audio Uniforms
| Uniform | Type | Range | Description |
|---------|------|-------|-------------|
| `bass` | `float` | 0.0 - 1.0 | Low frequency energy (20-250 Hz) |
| `mids` | `float` | 0.0 - 1.0 | Mid frequency energy (250-2000 Hz) |
| `highs` | `float` | 0.0 - 1.0 | High frequency energy (2000-20000 Hz) |
| `volume` | `float` | 0.0 - 1.0 | Overall volume level |
| `beat` | `float` | 0.0 - 1.0 | Beat detection (spikes on beats) |
| `audio_time` | `float` | 0.0+ | Accumulated time that speeds up with audio energy |

## Audio Reactivity Examples

### Basic Color Response
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // Color responds to frequency bands
    vec3 col = vec3(bass, mids, highs);

    fragColor = vec4(col, 1.0);
}
```

### Beat-Reactive Pulse
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Circle that pulses on beats
    float radius = 0.3 + beat * 0.2;  // Grows on beat
    float d = length(uv);
    float circle = smoothstep(radius, radius - 0.02, d);

    // Brightness flash on beat
    vec3 col = vec3(circle) * (1.0 + beat * 0.5);

    fragColor = vec4(col, 1.0);
}
```

### Audio-Driven Animation Speed
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    // audio_time speeds up with music energy
    float t = audio_time;

    // Pattern that animates faster with more audio energy
    float pattern = sin(uv.x * 10.0 + t) * sin(uv.y * 10.0 + t);

    fragColor = vec4(vec3(pattern), 1.0);
}
```

### Bass-Reactive Zoom
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Zoom in when bass hits
    float zoom = 1.0 + bass * 0.5;
    uv *= zoom;

    // Your pattern here...
    float pattern = sin(length(uv) * 20.0 - iTime * 2.0);

    fragColor = vec4(vec3(pattern), 1.0);
}
```

## Tweaking Existing Shaders

### Common Parameters to Adjust

#### Reactivity Strength
Look for multipliers on audio uniforms:
```glsl
// Less reactive
col *= 1.0 + beat * 0.2;

// More reactive
col *= 1.0 + beat * 1.0;
```

#### Speed
Look for time multipliers:
```glsl
// Slower
float t = audio_time * 0.5;

// Faster
float t = audio_time * 2.0;
```

#### Rotation Speed
```glsl
// Slower rotation
float rotSpeed = 0.05;

// Faster rotation
float rotSpeed = 0.2;

// Audio-reactive rotation
float rotSpeed = 0.1 + bass * 0.3;
```

#### Color Intensity
```glsl
// Subtle color response
baseCol.r += bass * 0.1;

// Strong color response
baseCol.r += bass * 0.5;
```

### Example: Modifying universe_within_reactive.glsl

Key parameters you can tweak:

```glsl
// Line 95-96: Rotation speed
float rotSpeed = 0.1 + (bass * 0.15) + (beat * 0.3);
//              ^^^    ^^^^^^^^^^^^^   ^^^^^^^^^^^^
//              base   bass influence  beat influence

// Line 102-103: Zoom/pulse amount
float beatPulse = 1.0 + beat * 0.3 + bass * 0.15;
//                      ^^^^^^^^^   ^^^^^^^^^^^
//                      beat zoom   bass zoom

// Line 106: Line thickness
float lineThickness = 1.0 + bass * 1.5 + beat * 0.5;
//                          ^^^^^^^^^   ^^^^^^^^^
//                          bass        beat

// Line 124-126: Color response
baseCol.r += bass * 0.4;   // Red from bass
baseCol.g += mids * 0.2;   // Green from mids
baseCol.b += highs * 0.4;  // Blue from highs

// Line 131: Brightness flash
col *= 1.0 + beat * 0.8 + bass * 0.3;
//           ^^^^^^^^^   ^^^^^^^^^
//           beat flash  bass flash
```

## Creating a New Shader

### Step 1: Start with a Template
Copy an existing shader or use this template:

```glsl
// My Custom Shader
// Audio uniforms: bass, mids, highs, volume, beat, audio_time

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Centered UV coordinates (-0.5 to 0.5 on Y axis)
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Animation time (speeds up with audio)
    float t = audio_time * 0.5;

    // Your visualization here
    vec3 col = vec3(0.0);

    // Example: spinning rings
    float d = length(uv);
    float ring = sin(d * 30.0 - t * 5.0);
    ring = smoothstep(0.0, 0.1, ring);

    // Audio-reactive color
    col = vec3(ring) * vec3(0.5 + bass * 0.5, 0.3 + mids * 0.3, 0.5 + highs * 0.5);

    // Beat flash
    col *= 1.0 + beat * 0.5;

    // Output
    fragColor = vec4(col, 1.0);
}
```

### Step 2: Save and Test
1. Save your file as `shaders/my_shader.glsl`
2. Restart the visualizer
3. Select "my_shader" from the preset dropdown
4. Test with different audio files

### Step 3: Iterate
- Adjust values
- Test again
- Repeat until it looks good!

## Tips for Good Audio Reactivity

### 1. Use Smoothing
Raw audio values can be jumpy. The `audio_time` uniform is pre-smoothed.

### 2. Layer Effects
Combine multiple audio responses:
```glsl
// Multiple effects layered
float effect = bass * 0.5 + beat * 0.3 + volume * 0.2;
```

### 3. Use Different Frequencies for Different Things
```glsl
// Bass for movement/size
float size = 1.0 + bass * 0.5;

// Mids for detail
float detail = 10.0 + mids * 20.0;

// Highs for sparkle/brightness
float brightness = 1.0 + highs * 0.3;
```

### 4. Beat Detection for Impact
The `beat` uniform spikes on drum hits - use it for dramatic effects:
```glsl
// Flash on beat
col *= 1.0 + beat * 0.8;

// Shake on beat
uv += vec2(sin(iTime * 50.0), cos(iTime * 50.0)) * beat * 0.01;
```

### 5. Avoid Over-Reactivity
Too much movement can be nauseating. Start subtle and increase.

## Porting Shaders from Shadertoy

Most Shadertoy shaders work directly! Just:

1. Copy the code
2. Save as `.glsl` file in `shaders/`
3. Add audio reactivity by using the audio uniforms

### Common Modifications
```glsl
// Replace Shadertoy's iTime with audio_time for audio-reactive speed
float t = iTime;        // Original
float t = audio_time;   // Audio-reactive

// Add color response
vec3 col = originalColor;
col.r += bass * 0.2;    // Add bass to red
col.b += highs * 0.2;   // Add highs to blue

// Add beat pulse
col *= 1.0 + beat * 0.3;
```

## Troubleshooting

### Shader won't compile
- Check for syntax errors (missing semicolons, unmatched braces)
- Ensure `mainImage` function signature is correct
- Look at the error message in the GUI

### No audio reactivity
- Make sure you're using the audio uniforms (`bass`, `mids`, `highs`, `beat`, `audio_time`)
- Check that the multipliers are large enough to see

### Animation too fast/slow
- Adjust time multipliers
- Use `audio_time * 0.5` for slower, `audio_time * 2.0` for faster

### Colors look wrong
- Check color values are in 0-1 range
- Use `clamp()` if needed: `col = clamp(col, 0.0, 1.0);`

## Resources

- [Shadertoy](https://www.shadertoy.com/) - Thousands of shader examples
- [The Book of Shaders](https://thebookofshaders.com/) - Learn GLSL
- [GLSL Sandbox](http://glslsandbox.com/) - Another shader playground
