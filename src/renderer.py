"""GPU renderer using moderngl for shader-based visualization."""

import moderngl
import numpy as np
from pathlib import Path


class ShaderRenderer:
    """Renders GLSL shaders with audio-reactive uniforms using GPU acceleration."""

    def __init__(self, width: int, height: int, use_gpu: bool = True):
        self.width = width
        self.height = height
        self.window = None

        if use_gpu:
            # Use pyglet hidden window for proper GPU acceleration
            self._init_gpu_context()
        else:
            # Fallback to standalone (software) context
            self.ctx = moderngl.create_standalone_context()

        # Create framebuffer for offscreen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        # Fullscreen quad vertices
        self.quad = self.ctx.buffer(
            np.array([
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ], dtype='f4').tobytes()
        )

        self.program = None
        self.vao = None

    def _init_gpu_context(self):
        """Initialize a GPU-accelerated context using pyglet."""
        import pyglet

        # Create a hidden window to get a proper GPU context
        config = pyglet.gl.Config(
            double_buffer=True,
            major_version=3,
            minor_version=3,
        )

        # Create minimal hidden window
        self.window = pyglet.window.Window(
            width=1, height=1,
            visible=False,
            config=config,
        )

        # Now create moderngl context from the active pyglet context
        self.ctx = moderngl.create_context()

        # Print GPU info
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")

    def load_shader(self, shader_path: str):
        """Load and compile a fragment shader."""
        vertex_shader = """
        #version 330
        in vec2 in_position;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """

        # Read fragment shader and add our header
        shader_code = Path(shader_path).read_text()

        # Convert Shadertoy-style to standard GLSL
        fragment_shader = self._convert_shadertoy(shader_code)

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.quad, '2f', 'in_position')],
        )

    def _convert_shadertoy(self, shader_code: str) -> str:
        """Convert Shadertoy shader to standard GLSL 330."""
        header = """
#version 330

// Standard uniforms
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;

// Audio uniforms
uniform float bass;
uniform float mids;
uniform float highs;
uniform float volume;
uniform float beat;
uniform float audio_time;

out vec4 _fragColor;
"""
        # Replace mainImage signature - rename parameter to avoid conflict with output
        shader_code = shader_code.replace(
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )",
            "void mainImage( out vec4 _outputColor, in vec2 fragCoord )"
        )
        shader_code = shader_code.replace(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord)",
            "void mainImage(out vec4 _outputColor, in vec2 fragCoord)"
        )

        # Replace fragColor assignments inside mainImage with _outputColor
        # This handles the common pattern at the end of mainImage
        shader_code = shader_code.replace("fragColor =", "_outputColor =")
        shader_code = shader_code.replace("fragColor=", "_outputColor =")

        # Add main() wrapper
        footer = """
void main() {
    mainImage(_fragColor, gl_FragCoord.xy);
}
"""
        return header + shader_code + footer

    def render_frame(self, time: float, frame: int, audio_data: dict) -> bytes:
        """Render a single frame and return as RGB bytes."""
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Set uniforms
        if 'iResolution' in self.program:
            self.program['iResolution'].value = (float(self.width), float(self.height))
        if 'iTime' in self.program:
            self.program['iTime'].value = time
        if 'iFrame' in self.program:
            self.program['iFrame'].value = frame

        # Audio uniforms
        if 'bass' in self.program:
            self.program['bass'].value = audio_data.get('bass', 0.0)
        if 'mids' in self.program:
            self.program['mids'].value = audio_data.get('mids', 0.0)
        if 'highs' in self.program:
            self.program['highs'].value = audio_data.get('highs', 0.0)
        if 'volume' in self.program:
            self.program['volume'].value = audio_data.get('volume', 0.0)
        if 'beat' in self.program:
            self.program['beat'].value = audio_data.get('beat', 0.0)
        if 'audio_time' in self.program:
            self.program['audio_time'].value = audio_data.get('audio_time', 0.0)

        # Render
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read pixels
        data = self.fbo.color_attachments[0].read()
        return data

    def render_frame_rgb(self, time: float, frame: int, audio_data: dict) -> np.ndarray:
        """Render a frame and return as numpy RGB array."""
        raw = self.render_frame(time, frame, audio_data)
        # Convert from RGBA to RGB, flip vertically
        img = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        img = np.flip(img, axis=0)  # Flip vertically
        return img[:, :, :3]  # Drop alpha

    def close(self):
        """Clean up OpenGL resources."""
        if self.vao:
            self.vao.release()
        if self.program:
            self.program.release()
        if self.fbo:
            self.fbo.release()
        if self.ctx:
            self.ctx.release()
        if self.window:
            self.window.close()
