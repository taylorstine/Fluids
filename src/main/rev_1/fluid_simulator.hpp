#ifndef _TJS_FLUID_SIMULATION
#define _TJS_FLUID_SIMULATION

struct FluidSimulator{

  FluidSimulator(int N, int M, float dt = 0.1,
		 float visc=1e-5, float diff=1.0e-3);
  ~FluidSimulator();
  void simulate();
  void reset();
  void add_density(int i, int j);
  void add_force(int vx, int vy, int i, int j);

  

  int N, M;
  float *U, *V, *U_0, *V_0, *D, *D_0;
  float dt, visc, diff;

private:
  void init();
  void clean_up();
  inline int AT(int,int);
  template<class T>
  inline void swap(T&, T&);
  void add_source(float *x, float* s, float dt);
  void diffuse(int b, float* x, float *x0, float diff, float dt);
  void advect(int b, float *d, float *d0, float *u, float *v, float dt);
  void set_boundary(int b, float *x);
  void step_density(float *x, float *x0, float *u, float *v, float diff, float dt);
  void step_velocity(float* u, float *v, float *u0, float *v0, float visc, float dt);
  void project(float *u, float *v, float *p, float *div);
};
#endif
