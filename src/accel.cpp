#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
              other.low_bnd[0] <= this->upper_bnd[0]) ||
             (this->low_bnd[0] >= other.low_bnd[0] &&
                 this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
              other.low_bnd[1] <= this->upper_bnd[1]) ||
             (this->low_bnd[1] >= other.low_bnd[1] &&
                 this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
              other.low_bnd[2] <= this->upper_bnd[2]) ||
             (this->low_bnd[2] >= other.low_bnd[2] &&
                 this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  
  // Get the inverse direction for efficiency
  Vec3f inv_dir = ray.safe_inverse_direction;
  
  Vec3f t0 = (low_bnd - ray.origin) * inv_dir;   
  Vec3f t1 = (upper_bnd - ray.origin) * inv_dir;  
  
  Vec3f t_min = Min(t0, t1);  // Entry times for each axis
  Vec3f t_max = Max(t0, t1);  // Exit times for each axis
  
  Float t_near = ReduceMax(t_min);
  Float t_far = ReduceMin(t_max);
  
  if (t_near > t_far || t_far < 0) {
    return false;  // No intersection
  }
  
  *t_in = t_near;
  *t_out = t_far;
  
  return true;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
    const ref<TriangleMeshResource> &mesh, SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType    = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0  = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1  = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2  = Cast<InternalScalarType>(vertices[v_idx[2]]);

  
  InternalVecType origin = Cast<InternalScalarType>(ray.origin);
  
  InternalVecType edge1 = v1 - v0;
  InternalVecType edge2 = v2 - v0;
  

  InternalVecType pvec = Cross(dir, edge2);
  InternalScalarType det = Dot(edge1, pvec);
  
  constexpr InternalScalarType epsilon = InternalScalarType(1e-8);
  if (std::abs(det) < epsilon) {
    return false;  // Ray is parallel to triangle
  }
  
  InternalScalarType inv_det = InternalScalarType(1) / det;
  
  InternalVecType tvec = origin - v0;
  
  InternalScalarType u = Dot(tvec, pvec) * inv_det;
  

  if (u < InternalScalarType(0) || u > InternalScalarType(1)) {
    return false;
  }
  
  InternalVecType qvec = Cross(tvec, edge1);
  InternalScalarType v = Dot(dir, qvec) * inv_det;
  
  if (v < InternalScalarType(0) || u + v > InternalScalarType(1)) {
    return false;
  }
  InternalScalarType t = Dot(edge2, qvec) * inv_det;
  
  if (t < static_cast<InternalScalarType>(ray.t_min) || t > static_cast<InternalScalarType>(ray.t_max)) {
    return false;
  }

  // We will reach here if there is an intersection

  CalculateTriangleDifferentials(interaction,
      {static_cast<Float>(1 - u - v), static_cast<Float>(u),
          static_cast<Float>(v)},
      mesh, triangle_index);
  AssertNear(interaction.p, ray(t));
  assert(ray.withinTimeRange(t));
  ray.setTimeMax(t);
  return true;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
      Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd   = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh  = mesh;   // set the pointer
  this->bound = bound;  // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const {
  return bound;
}

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END