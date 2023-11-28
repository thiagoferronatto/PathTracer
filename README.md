# A rudimentary C++ path tracer

It's just a huge header file. To use it, include it in your project and make sure you have a functioning Vector class akin to GLM's vec2/3/4.

The `render` function returns an array of quadruplets of bytes representing RGBA colors. Just output that to whatever medium you'd like:

- directly to the screen with OpenGL or another graphics API,
- out into an image file with a separate image lib, etc.

## An example of what it can do

![image](https://github.com/thiagoferronatto/path-tracer/assets/31262053/023dbfc2-ead9-4380-855f-928774ea263b)
