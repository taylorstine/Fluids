#include "mesh.hpp"
#define ASSERT(condition){if(!(condition)){std::cerr<<"ASSERTION FAILED: "<<#condition<<"@"<<__FILE__<<"("<<__LINE__<<")"<<std::endl;}}

int num_pts;
inline int AT(int i, int j){return i+(num_pts+2)*j;}
bool create_mesh(Mesh& mesh, int N, int M, int Width, int Height){
  mesh.N = N;
  mesh.M = M;
  num_pts = N;

  mesh.num_vertices = (N+2)*(M+2);
  mesh.vertices = new(std::nothrow) Vertex[mesh.num_vertices];
  ASSERT(mesh.vertices !=NULL);
  float dx = (1.0)/(N+1)* (GLfloat)Width/(GLfloat)Height;
  float dy = (1.0)/(M+1); 
  for(int i = 0; i < N+2; i++){
    for(int j = 0; j < M+2; j++){
      ASSERT(AT(i,j)<mesh.num_vertices);
      mesh.vertices[AT(i,j)].x = dx * i;
      mesh.vertices[AT(i,j)].y = dy*j;
    }
  }
  
  mesh.num_triangles = (N+1)*(M+1)*2;
  mesh.triangles = new (std::nothrow)Triangle[mesh.num_triangles];
  ASSERT(mesh.triangles != NULL);
  int tri_idx = 0;
  for(int i = 0; i < N+1; i++){
    for(int j = 0; j < M+1; j++){
      ASSERT(AT(i,j)<mesh.num_vertices);
      mesh.triangles[tri_idx].indices[0] = AT(i,j);
      mesh.triangles[tri_idx].indices[1] = AT(i+1,j);
      mesh.triangles[tri_idx].indices[2] = AT(i+1,j+1);
      tri_idx++;
      mesh.triangles[tri_idx].indices[0] = AT(i,j);
      mesh.triangles[tri_idx].indices[1] = AT(i+1,j+1);
      mesh.triangles[tri_idx].indices[2] = AT(i,j+1);
      tri_idx++;
    }
  }

  mesh.point_indices = new (std::nothrow)GLuint[mesh.num_vertices];
  ASSERT(mesh.point_indices !=NULL);
  for(int i = 0; i < mesh.num_vertices; i++)
    mesh.point_indices[i] = i;

  mesh.colors = new Color[mesh.num_vertices];
  for(int i = 0; i < N+2; i++){
    for(int j = 0; j < M+2; j++){
      mesh.colors[AT(i,j)].r = 0.0;
      mesh.colors[AT(i,j)].g = 0.0;
      mesh.colors[AT(i,j)].b = 0.0;
      mesh.colors[AT(i,j)].a = 1.0;
    }
  }

  return true;
}
