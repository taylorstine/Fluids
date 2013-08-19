#ifndef _TJS_MESH_
#define _TJS_MESH_
#include <GL/gl.h>
#include <math/math.hpp>
#include <new>
#include <iostream>

struct Color{
  float r;
  float g;
  float b;
  float a;
};
struct Vertex{
  float x;
  float y;
};

struct Triangle{
  GLuint indices[3];
};

struct Mesh{
  int N;
  int M;
  Vertex * vertices;
  Triangle * triangles;
  Color *colors;
  GLuint * point_indices;
  int num_vertices;
  int num_triangles;

};

bool create_mesh(Mesh& mesh, int N, int M, int Width, int Height);

#endif
