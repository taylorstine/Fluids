#ifndef _TJS_FLUID_SIMULATION_
#define _TJS_FLUID_SIMULATION_
#include <vector>

struct Source{
  int i;
  int j;
  float value;
};

struct FluidSimulator{
  float * U_0, *V_0, *D_0;
  float *U, *V, *D;

  float dt;
  int N;
  int M;
  float viscosity;
  float diffusion_rate;


  FluidSimulator(int N, int M, float dt, float viscocity, float diffusion_rate);
  ~FluidSimulator();
  void simulate();
  void reset();
  void clean_up();
  //  void add_density_source(int i, int j, float value);
  //  void add_u_source(int i, int j, float value);
  //  void add_v_source(int i, int j, float value);
private:
  typedef std::vector<Source> SourceList;
  SourceList d_sources;
  SourceList u_sources;
  SourceList v_sources;
  void set_sources(float *d, float *u, float *v);
  void set_boundary(int b, float *x);
  void add_source(float* x, float * s);
  void diffusion(int boundary, float*x,
		 float*x0, float difffusion_rate);
  void advection(int boundary, float* d, float* d0,
		 float *u, float *v);
  void project(float* u, float* v, float *p, float *div);
  void step_density(float* x, float* x0, float* u, float* v,
		    float diffusion_rate);
  void step_velocity(float *u, float *v, float *u0, float *v0,
		     float viscosity);


  inline int AT(int i, int j){return i+(N+2)*j;}
  template<class T>
  inline void swap(T& a,T& b){ T c = a; a = b; b = c;}


};

#endif
