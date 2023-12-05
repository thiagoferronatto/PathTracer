# A rudimentary C++ path tracer

It's just a huge header file. To use it, include it in your project and make sure you have a functioning Vector class akin to GLM's vec2/3/4.

The `render` function returns an array of quadruplets of bytes representing RGBA colors. Just output that to whatever medium you'd like:

- directly to the screen with OpenGL or another graphics API,
- out into an image file with a separate image lib, etc.

## An example of what it can do

![287806732-2390ecab-025c-4a6a-8646-878f5fee590f](https://github.com/thiagoferronatto/PathTracer/assets/31262053/4a9b25ec-e693-49dd-b678-cea2d20e4482)
