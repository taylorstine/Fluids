#include "fluid_simulator.hpp"
#include <iostream>

FluidSimulator::FluidSimulator(int N, int M, float dt,
			       float viscosity,
			       float diffusion_rate):U(NULL),
						     V(NULL),
						     D(NULL),
						     U_0(NULL),
						     V_0(NULL),
						     D_0(NULL)
{
  this->N = N;
  this->M = M;
  this->dt = dt;
  this->viscosity = viscosity;
  this->diffusion_rate = diffusion_rate;
  reset();
}
FluidSimulator::~FluidSimulator(){
  clean_up();
}

void FluidSimulator::clean_up(){
  delete [] U;
  delete [] V;
  delete [] D;
  delete [] V_0;
  delete [] U_0;
  delete [] D_0;

}

void FluidSimulator::reset(){
  this->U = new float[(N+2)*(M+2)];
  this->V = new float[(N+2)*(M+2)];
  this->D = new float[(N+2)*(M+2)];
  this->U_0 = new float[(N+2)*(M+2)];
  this->V_0 = new float[(N+2)*(M+2)];
  this->D_0 = new float[(N+2)*(M+2)];
  for(int i = 0; i < N+2; i++){
    for(int j =0; j< M+2; j++){
      this->U_0[AT(i,j)] = 0.0;
      this->V_0[AT(i,j)] = 0.0;
      this->D_0[AT(i,j)] = 0.2;
      this->U[AT(i,j)] = 0.0;
      this->V[AT(i,j)] = 0.0;
      this->D[AT(i,j)] = 0.0;
    }
  }
}

void FluidSimulator::set_boundary(int b, float *x){
  
  for(int i= 0; i<N+1;i++){
    x[AT(0,i)] = b==1 ? -x[AT(1,i)] : x[AT(1,i)];
    x[AT(N+1,i)] = b==1 ? 2*100-x[AT(N,i)] : x[AT(N,i)];
    x[AT(i,0)] = b==2 ? -x[AT(i,1)] : x[AT(1,i)];
    x[AT(i,M+1)] = b==2 ? -x[AT(i,M)] : x[AT(i,M)];
  }
  x[AT(0,0)] = 0.5 * (x[AT(1,0)] + x[AT(0,1)]);
  x[AT(0,M+1)] = 0.5 * (x[AT(1,M+1)] + x[AT(0,M)]);
  x[AT(N+1,0)] = 0.5 * (x[AT(N,0)] + x[AT(N+1, 1)]);
  x[AT(N+1, M+1)] = 0.5 * (x[AT(N,M+1)] + x[AT(N+1,M)]);
  }

/*
void FluidSimulator::set_boundary(int b, float *x){
  for(int i =0; i<N+2; i++){
    U[AT(i,1)] = -U[AT(i,0)];
    U[AT(i,M+1)] = 2*10-U[AT(i,M)];
  }
  for(int j = 0; j<M+2; j++){
    V[AT(1,j)] = -V[AT(0,j)];
    V[AT(N+1,j)] = -V[AT(N,j)];
  }
}
*/

void FluidSimulator::add_source(float *x, float *s){
  float dt = FluidSimulator::dt;
  int size = (N+2)*(M+2);
  for(int i = 0; i < size; i++){
    x[i] += dt * s[i];
  }
}


void FluidSimulator::diffusion(int boundary, float* x, float *x0,
			       float diffusion_rate){
  float dt = FluidSimulator::dt;
  float a= dt *diffusion_rate*N*M;
  //use Gauss-Seidel relaxation to ensure this won't explode
  for(int k = 0; k<20; k++){
    for(int i = 1; i <N+1; i++){
      for(int j = 1; j<M+1; j++){
	x[AT(i,j)] = (x0[AT(i,j)] + a*(x[AT(i-1,j)] + x[AT(i+1,j)]+
				       x[AT(i,j-1)] + x[AT(i,j+1)]))/(1+4*a);
      }
    }
    set_boundary(boundary, x);
  }
}

void FluidSimulator::advection(int boundary, float *d, float *d0,
			       float *u, float *v){
  int i0=0, j0=0, i1=0, j1=1;
  float dt = FluidSimulator::dt;
  float x, y, dt0;
  float s0=0.0, s1=0.0, t0=0.0, t1=0.0;
  for(int i = 1; i <N+1; i++){
    for(int j = 1; j<M+1; j++){
      x = i-dt0*u[AT(i,j)];
      y = j-dt0*v[AT(i,j)];
      //clamp
      if(x<0.5)	x = 0.5;
      if(x>N+.5) x=N+.5;
      if(y<.5) y=.5;
      if(y>N+.5) y = N+.5;
      i0 = (int)x;
      j0 = (int)y;
      i1 = i0+1;
      j1 = j0+1;
      //find interpolation values
      s1 = x-i0; s0 = 1-s1;
      t1 = y-j0; t0 = 1-t1;
      d[AT(i,j)] = s0*(t0*d0[AT(i0,j0)] + t1*d0[AT(i0,j1)])+
	s1 *(t0*d0[AT(i1,j0)] + t1*d0[AT(i1,j1)]);
    }
  }
  set_boundary(boundary, d);

}

void FluidSimulator::project(float *u, float *v, float *p, float *div){
  float h = 1.0/N;
  for(int i =1; i <N+1; i++){
    for(int j = 1; j<M+1; j++){
      div[AT(i,j)] = -.5*h*(u[AT(i+1,j)] - u[AT(i-1,j)]+
			    v[AT(i,j+1)] - v[AT(i,j-1)]);
      p[AT(i,j)] = 0;
    }
  }
  set_boundary(0, div);
  set_boundary(0, p);
  
  for(int k = 0; k < 20; k++){
    for(int i = 1; i <N+1; i++){
      for(int j = 1; j<M+1; j++){
	p[AT(i,j)] = (div[AT(i,j)] + p[AT(i-1,j)] + p[AT(i+1,j)]+
		      p[AT(i,j-1)] + p[AT(i,j+1)])/4.0;
      }
    }
    set_boundary(0, p);
  }
  for(int i=1; i<N+1; i++){
    for(int j=1; j<M+1;j++){
      u[AT(i,j)] -= 0.5*(p[AT(i+1,j)] - p[AT(i-1,j)])/h;
      v[AT(i,j)] -= 0.5*(p[AT(i,j+1)] - p[AT(i,j-1)])/h;
    }
  }
  set_boundary(1, u);
  set_boundary(2, v);

}

template<class T>
inline void swap(T &a, T &b){ T c = a; a = b; b = c;}

void FluidSimulator::step_density(float *x, float *x0, float*u,
				  float *v, float diffusion_rate){
  add_source(x, x0);
  swap(x0, x); diffusion(0, x, x0, diffusion_rate);
  swap(x0, x); advection(0, x, x0, u, v);
}

void FluidSimulator::step_velocity(float *u, float *v, float *u0,
				   float *v0, float viscosity){
  add_source(u,u0);
  add_source(v,v0);
  swap(u0,u); diffusion(1, u, u0, viscosity);
  swap(v0,v); diffusion(2, v, v0, viscosity);
  project(u,v,u0,v0);
  swap(u0,u); swap(v0,v);
  advection(1,u,u0,u0,v0);
  advection(2,v,v0,u0,v0);
  project(u,v,u0,v0);
}

void FluidSimulator::set_sources(float *d, float* u, float *v){
  //  d[AT(50,60)] = 10.0;
  //  for(int i = 0; i < N+1; i++){
  //    u[AT(i,M-2)] = 10;
  //  }
      //  u[AT(,50)] = 1;
  //  v[AT(50,50)] = 0.0;
}

void FluidSimulator::simulate(){

  set_sources(D_0, U_0, V_0);
  step_velocity(U, V, U_0, V_0, viscosity);
  step_density(D, D_0, U, V, diffusion_rate);
  
}

