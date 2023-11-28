#ifndef RAY_TRACER_HPP
#define RAY_TRACER_HPP

#include <optional>
#include <thread>
#include <vector>

#include "math/glm/glm.hpp"

// a bad (but kinda fast) way to generate a pseudorandom float between 0 and 1
static constexpr auto invRandMax{1.0f / float(RAND_MAX)};
#define RANDOM (invRandMax * float(rand()))

namespace rt {

using namespace glm;

using IntersectionData = std::optional<std::tuple<vec3, vec3>>;

class Ray {
public:
  Ray(const vec3 &position, const vec3 &direction)
      : _position{position}, _direction{normalize(direction)} {}

  [[nodiscard]] vec3 origin() const { return _position; }

  void setPosition(vec3 position) { _position = position; }

  [[nodiscard]] vec3 direction() const { return _direction; }

  void setDirection(vec3 direction) { _direction = normalize(direction); }

private:
  vec3 _position, _direction;
};

struct Material {
  Material() : diffuse{1.0f}, specular{1.0f}, roughness{0.1f}, metallic{} {}

  Material(const vec3 &diffuse, const vec3 &specular, float roughness,
           float metallic)
      : diffuse{diffuse}, specular{specular}, roughness{roughness},
        metallic{metallic} {}

  vec3 diffuse, specular;
  float roughness, metallic;
};

struct Object {
  [[nodiscard]] vec3 position() const { return _transform[3]; }

  virtual void rotate(const vec3 &axis, float angle) {
    _transform = glm::rotate(_transform, angle, normalize(axis));
  }

  virtual void translate(const vec3 &xyz) {
    _transform = glm::translate(_transform, xyz);
  }

  virtual void scale(const vec3 &xyz) {
    _transform = glm::scale(_transform, xyz);
  }

protected:
  mat4 _transform{identity<mat4>()};
};

struct Camera : public Object {
  void scale(const vec3 &xyz) override {
    // no scaling for cameras
  }
};

struct Shape : public Object {
  explicit Shape(const Material &material) : material{material} {}

  [[nodiscard]] virtual IntersectionData intersection(const Ray &ray) const = 0;

  Material material;
};

struct Sphere : public Shape {
  Sphere(const vec3 &center, float radius, const Material &material = {})
      : Shape{material}, center{center}, radius{radius} {}

  [[nodiscard]] IntersectionData intersection(const Ray &ray) const override {
    // OC should, by all means, be C - O, not O - C, but somehow it doesn't work
    // like that
    auto C{center}, O{ray.origin()}, v{ray.direction()}, OC{O - C};
    auto r{radius};
    auto ocDotV{dot(OC, v)};
    auto delta{ocDotV * ocDotV - (dot(OC, OC) - r * r)};
    if (delta < 0.0f) // ray doesn't hit the sphere at all
      return std::nullopt;
    auto t{-ocDotV - std::sqrt(delta)};
    if (t <= 0.0f) // ray hits sphere behind the camera
      return std::nullopt;
    auto p{O + t * v};
    return std::forward_as_tuple(p, normalize(p - C));
  }

  vec3 center;
  float radius;
};

struct Plane : public Shape {
  Plane(const vec3 &point, const vec3 &normal, const Material &material = {})
      : Shape{material}, point{point}, normal{normalize(normal)} {}

  [[nodiscard]] IntersectionData intersection(const Ray &ray) const override {
    auto P{point}, n{normal}, O{ray.origin()}, v{ray.direction()}, OP{P - O};
    auto vDotN{dot(v, n)};
    if (fabsf(vDotN) < 1e-5) // ray parallel to plane
      return std::nullopt;
    auto t{dot(OP, n) / vDotN};
    if (t <= 0.0f) // ray hits plane behind the camera
      return std::nullopt;
    return std::forward_as_tuple(O + t * v, n);
  }

  vec3 point, normal;
};

struct Quad : public Shape {
  Quad(const vec3 &A, const vec3 &B, const vec3 &C, const vec3 &D,
       const Material &material = {})
      : Shape{material}, A{A}, B{B}, C{C}, D{D} {}

  [[nodiscard]] IntersectionData intersection(const Ray &ray) const override {
    auto AB{B - A}, AD{D - A};
    auto n{cross(AB, AD)}, O{ray.origin()}, v{ray.direction()}, OA{A - O};
    auto vDotN{dot(v, n)};
    if (fabsf(vDotN) < 1e-5)
      return std::nullopt;
    auto t{dot(OA, n) / vDotN};
    if (t < 1e-5) // why does 0.0f fuck this up?
      return std::nullopt;
    auto P{O + t * v};
    auto BC{C - B}, CD{D - C}, PA{A - P}, PB{B - P}, PC{C - P}, PD{D - P};
    auto den{length(cross(AB, AD)) + length(cross(BC, CD))};
    float a{length(cross(PA, PB)) / den}, b{length(cross(PB, PC)) / den},
        c{length(cross(PC, PD)) / den}, d{length(cross(PA, PD)) / den};
    if (a < 0.0f || a > 1.0f || //
        b < 0.0f || b > 1.0f || //
        c < 0.0f || c > 1.0f || //
        d < 0.0f || d > 1.0f)
      return std::nullopt;
    return std::forward_as_tuple(P, n);
  }

  vec3 A, B, C, D;
};

struct PointLight {
  vec3 color, position;
  float intensity;
};

using Light = PointLight;

class Scene {
public:
  explicit Scene(Camera camera) : _camera{std::move(camera)} {}

  void addShape(const Shape *shape) { _shapes.push_back(shape); }

  void addLight(const Light &light) {
    _lights.push_back(std::make_shared<Light>(light));
  }

  [[nodiscard]] const auto &camera() const { return _camera; }

  [[nodiscard]] const auto &shapes() const { return _shapes; }

  [[nodiscard]] const auto &lights() const { return _lights; }

private:
  std::vector<const Shape *> _shapes;
  std::vector<std::shared_ptr<Light>> _lights;
  Camera _camera;
};

class RayTracer {
public:
  explicit RayTracer(u64vec2 resolution, float projectionZ = -0.5f)
      : _resolution{resolution},
        _projectionPlane{0.001f * vec2{resolution}, projectionZ},
        _pixelDimensions{_projectionPlane.x / float(resolution.x),
                         float(_projectionPlane.y) / float(resolution.y)} {}

  [[nodiscard]] static vec3 tracePath(vec3 position, vec3 direction,
                                      const Scene &scene, size_t maxBounces = 8,
                                      size_t indirectSamples = 4) {
    if (maxBounces == 0)
      return {};
    auto nextMaxBounces{maxBounces - 1};
    vec3 rv{RANDOM - 0.5f, RANDOM - 0.5f, RANDOM - 0.5f};
    Ray ray{position, direction};
    for (auto shape : scene.shapes()) {
      if (auto intersection{shape->intersection(ray)};
          intersection.has_value()) {
        auto [p, n]{intersection.value()};
        auto metallicCompl{1 - shape->material.metallic};
        vec3 color{}; // _ambient * shape->material.diffuse
        for (const auto &light : scene.lights()) {
          auto l{light->position - p};
          Ray lightRay{p, l + rv * 0.1f};
          bool shadow{};
          for (auto shadowShape : scene.shapes()) {
            if (auto shadowIntersection{shadowShape->intersection(lightRay)};
                shadowIntersection.has_value()) {
              shadow = true;
              break;
            }
          }
          if (shadow)
            continue;
          auto invSqDist{1.0f / dot(l, l)};
          l = normalize(l);
          color += metallicCompl * max(dot(n, l), 0.0f) * invSqDist *
                   light->intensity * light->color * shape->material.diffuse;
        }
        vec3 indirectDiffuse{};
        for (int i{}; i < indirectSamples; ++i) {
          // TODO: randomize two angles, theta and phi, and rotate by them
          // this would simulate a hemisphere around P given by n
          // TODO: do that in a way that doesn't tank performance
          auto dr{n + vec3{RANDOM - 0.5f, RANDOM - 0.5f, RANDOM - 0.5f}};
          auto tracedIndirect{tracePath(p, dr, scene, nextMaxBounces)};
          if (tracedIndirect != _ambient)
            indirectDiffuse += _indirectDampeningFactor *
                               shape->material.diffuse * tracedIndirect;
        }
        color += indirectDiffuse / float(indirectSamples);
        vec3 v, r;
        if (shape->material.roughness == 1.0f) // exactly 1 means no specular
          goto endRecursion;
        v = -ray.direction();
        r = 2.0f * dot(n, v) * n - v;
        r += shape->material.roughness * rv;
        color += _specularDampeningFactor * shape->material.specular *
                 tracePath(p, r, scene, nextMaxBounces);
      endRecursion:
        return clamp(color, _ambient, vec3{1.0f});
      }
    }
    return _ambient;
  }

  void render(const Scene &scene, u8vec4 *buffer, size_t samples = 4) const {
    static const auto negHalfProjectionPlane{-0.5f * _projectionPlane};
    const auto samplesPerAxis{std::ceil(std::sqrt(float(samples)))};
    const auto subPixelDimensions{_pixelDimensions / samplesPerAxis};
    for (size_t l{}; l < size_t(samplesPerAxis); ++l) {
      vec2 subPixelOffset{0, float(l) * subPixelDimensions.y};
      for (size_t m{}; m < size_t(samplesPerAxis); ++m) {
        subPixelOffset.x = float(m) * subPixelDimensions.x;
        for (size_t i{}, k{}; i < _resolution.y; ++i, k += _resolution.x) {
          auto py{negHalfProjectionPlane.y + float(i) * _pixelDimensions.y};
          for (size_t j{}; j < _resolution.x; ++j) {
            auto px{negHalfProjectionPlane.x + float(j) * _pixelDimensions.x};
            vec3 rayDirection{px + subPixelOffset.x, //
                              py + subPixelOffset.y, //
                              _projectionPlane.z};
            auto color{tracePath({}, rayDirection, scene)};
            color = 255.0f * min(max(color, _ambient), 1.0f);
            // super naïve blending method
            static constexpr float memory{0.75f}, oneMinusMemory{1 - memory};
            if (l != 0 || m != 0)
              color = memory * vec3(buffer[j + k]) + oneMinusMemory * color;
            // TODO: tone mapping, forget clamping
            buffer[j + k] = {u8vec3{color}, 255};
          }
        }
      }
    }
  }

private:
  static constexpr vec3 _ambient{0.05f};
  static constexpr float _indirectDampeningFactor{0.6f};
  static constexpr float _specularDampeningFactor{0.6f};

  u64vec2 _resolution;
  vec3 _projectionPlane; ///< (planeWidth, planeHeight, planeZ)
  vec2 _pixelDimensions;
};

} // namespace rt

#endif // RAY_TRACER_HPP
