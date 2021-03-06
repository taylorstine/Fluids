#ifndef _TJS_FLUID_SIMULATION_
#define _TJS_FLUID_SIMULATION_
#include <vector>
#include "math/math.hpp"

struct Source{
  int i;
  int j;
  float value;
};

enum Wall{TOP, BOTTOM, LEFT, RIGHT};
enum BC{NO_SLIP, FREE_SLIP, OUTFLOW, INFLOW};
enum Flow_Conditions{DRIVEN, FLOW_X, FLOW_Y};
struct FluidSimulator{

  float wall_velocity[4];
  int boundary_condition[4];
  int flow_condition;
  float flow_velocity;
  float ui, vi, pi;


  float * U_0, *V_0, *D_0;
  float *U, *V, *D;
  float *Fn, *Gn;
  float *Gx, *Gy;
  float *RHS;
  float *P, *P_0;
  float *R;
  
  int max_iters;
  float tau;
  float w;
  float TOL;
  float dt;
  float dx;
  float dy;
  float Re;
  float gamma;
  int N;
  int M;
  float viscosity;
  float diffusion_rate;


  FluidSimulator(int N, int M, float dt, float viscocity, float diffusion_rate);
  ~FluidSimulator();
  void simulate();
  void reset();
  void init();
  void clean_up();

private:
  typedef std::vector<Source> SourceList;
  SourceList d_sources;
  SourceList u_sources;
  SourceList v_sources;
  void set_sources(float *d, float *u, float *v);
  //  void set_boundary(int b, float *x);
  void add_source(float* x, float * s);
  void diffusion(int boundary, float*x,
		 float*x0);
  void advection(int boundary, float* d, float* d0,
		 float *u, float *v);
  //  void project(float* u, float* v, float *p, float *div);
  void step_density(float* x, float* x0, float* u, float* v);
  void step_velocity(float *u, float *v,
		     float* fn, float *gn,
		     float *p, float *p_0);
  void set_velocity_boundary(float * u, float *v, float *p);
  void set_density_boundary(float *d);
  inline float d2x(const float *f, int i, int j);
  inline float d2y(const float *f, int i, int j);
  inline float du2dx(const float*u, int i, int j);
  inline float duvdy(const float *u, const float *v, int i, int j);
  inline float duvdx(const float *u, const float *v, int i, int j);
  inline float dv2dy(const float *v, int i, int j);
  inline int eE(int i);
  inline int eW(int i);
  inline int eN(int j);
  inline int eS(int j);


  inline int AT(int i, int j){return i+(N+2)*j;}
  template<class T>
  inline void swap(T& a,T& b){ T c = a; a = b; b = c;}


};

#endif
