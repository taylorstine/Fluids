#include "fluid_simulator.hpp"
#include <iostream>

#ifdef OPENMP
#include <omp.h>
#endif

#define MAX_THREADS 20

float FluidSimulator::wall_velocity[4] = {0.0, 1.0, 0.0, 0.0};
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
  this->dt = .02;
  this->viscosity = viscosity;
  this->diffusion_rate = diffusion_rate;

  this->U = new float[(N+2)*(M+2)];
  this->V = new float[(N+2)*(M+2)];
  this->D = new float[(N+2)*(M+2)];
  this->D_0 = new float[(N+2)*(M+2)];
  this->P = new float[(N+2)*(M+2)];
  this->P_0 = new float[(N+2)*(M+2)];
  this->Fn = new float[(N+2)*(M+2)];
  this->Gn = new float[(N+2)*(M+2)];
  this->Gx = new float[(N+2)*(M+2)];
  this->Gy = new float[(N+2)*(M+2)];
  this->RHS = new float[(N+2)*(M+2)];
  this->R = new float[(N+2)*(M+2)];

  this->max_iters = 20;
  this->tau = .5;
  this->w = 1.7;
  this->Re = 1000;
  this->TOL = .01;
  this->dx = 1.0/(N+1);
  this->dy = 1.0/(M+1);
  this->gamma = .9;
  
#ifdef OPENMP
  omp_set_num_treads(MAX_THREADS);
#endif

  reset();
}
FluidSimulator::~FluidSimulator(){
  clean_up();
}

inline float FluidSimulator::d2x(const float * f, int i, int j){
  return (f[AT(i+1,j)]-2*f[AT(i,j)]+f[AT(i-1,j)])/pow(dx,2);
}
inline float FluidSimulator::d2y(const float * f, int i, int j){
  return (f[AT(i,j+1)]-2*f[AT(i,j)]+f[AT(i,j-1)])/pow(dy,2);
}
inline float FluidSimulator::du2dx(const float * u, int i, int j){
  return (1/dx * (pow( ( u[AT( i , j )] + u[AT( i+1 , j )])/2,2) -
		 pow( ( u[AT( i-1 , j )] + u[AT( i , j )] )/2,2)) + 
    gamma/dx * (fabs( u[AT( i , j )] + u[AT( i+1 , j )])/2 *
		(u[AT( i , j )] - u[AT( i+1 , j )])/2 -
		fabs( u[AT( i-1 , j )] + u[AT( i , j )] )/2 *
		(u[AT( i-1 , j )] - u[AT( i , j )])/2));
}

inline float FluidSimulator::duvdy(const float *u, const float *v,
				   int i, int j){
  return (1/dy *((v[AT( i , j )] + v[AT( i+1 , j )])/2 *
		(u[AT( i , j )] + u[AT( i , j+1 )])/2 -
		(v[AT( i , j-1 )] + v[AT( i+1 , j-1 )])/2 *
		(u[AT( i , j-1 )] + u[AT( i , j )])/2) +
    gamma/dy *(fabs( v[AT( i , j )] + v[AT( i+1 , j )] )/2 *
	       (u[AT( i , j )] - u[AT( i , j+1 )])/2 -
	       fabs( v[AT( i , j-1 )] + v[AT( i , j )])/2 *
	       (u[AT( i , j-1 )] - u[AT( i , j )])/2));
}

inline float FluidSimulator::duvdx(const float *u, const float *v,
				   int i, int j){
  return (1/dx *((v[AT( i , j )] + v[AT( i+1 , j )])/2 *
		(u[AT( i , j )] + u[AT( i , j+1 )])/2 -
		(v[AT( i-1 , j )] + v[AT( i , j )])/2 *
		(u[AT( i-1 , j )] + u[AT( i-1 , j+1 )])/2) +
	  gamma/dx *(fabs( u[AT(i,j)] + u[AT(i,j+1)] )/2 *
		     (v[AT(i,j)] - v[AT(i+1,j)])/2 -
		     fabs(u[AT(i=1,j)] + u[AT(i-1,j+1)])/2 *
		     (v[AT(i-1,j)] - v[AT(i,j)])/2));
}

inline float FluidSimulator::dv2dy(const float * v, int i, int j){
  return (1/dy * (pow((v[AT(i,j)]+v[AT(i,j+1)])/2,2) -
		 pow((v[AT(i,j-1)]+v[AT(i,j)])/2,2)) + 
    gamma/dy * (fabs(v[AT(i,j)] + v[AT(i,j+1)])/2 *
		(v[AT(i,j)] - v[AT(i,j+1)])/2 -
		fabs(v[AT(i,j-1)] + v[AT(i,j)])/2 *
		(v[AT(i,j-1)] - v[AT(i,j)])/2));
}

inline int FluidSimulator::eE(int i){return i<N? 1:0;}
inline int FluidSimulator::eW(int i){return i==0? 0:1;}
inline int FluidSimulator::eN(int j){return j<M? 1:0;}
inline int FluidSimulator::eS(int j){return j==0? 0:1;}


void FluidSimulator::clean_up(){
  delete [] U;
  delete [] V;
  delete [] D;
  delete [] D_0;
  delete [] P;
  delete [] P_0;
  delete [] Fn;
  delete [] Gn;
  delete [] Gx;
  delete [] Gy;
  delete [] RHS;
  delete [] R;
}

void FluidSimulator::reset(){
#pragma omp parallel for
  for(int i = 0; i < N+2; i++){
#pragma omp parallel for
    for(int j =0; j< M+2; j++){
      this->U[AT(i,j)] = 0.0;
      this->V[AT(i,j)] = 0.0;
      this->D[AT(i,j)] = 0.0;
      this->D_0[AT(i,j)] = 0.0;
      this->P[AT(i,j)] = 0.0;
      this->P_0[AT(i,j)] = 0.0;
      this->Fn[AT(i,j)] = 0.0;
      this->Gn[AT(i,j)] = 0.0;
      this->Gx[AT(i,j)] = 0.0;
      this->Gy[AT(i,j)] = 0.0;
      this->RHS[AT(i,j)] = 0.0;
      this->R[AT(i,j)] = 0.0;
    }
  }
}

/*void FluidSimulator::set_boundary(int b, float *x){
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
}*/

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
    //    set_boundary(boundary, x);
  }
}

void FluidSimulator::advection(int boundary, float *d, float *d0,
			       float *u, float *v){
  int i0=0, j0=0, i1=0, j1=1;
  float dt = FluidSimulator::dt;
  float x, y, dt0;
  float s0=0.0, s1=0.0, t0=0.0, t1=0.0;
#pragma omp parallel for
  for(int i = 1; i <N+1; i++){
#pragma omp parallel for
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
  //  set_boundary(boundary, d);

}

/*void FluidSimulator::project(float *u, float *v, float *p, float *div){
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

  }*/

template<class T>
inline void swap(T &a, T &b){ T c = a; a = b; b = c;}

void FluidSimulator::step_density(float *x, float *x0, float*u,
				  float *v, float diffusion_rate){
  add_source(x, x0);
  swap(x0, x); diffusion(0, x, x0, diffusion_rate);
  swap(x0, x); advection(0, x, x0, u, v);
}

void FluidSimulator::step_velocity(float *u, float *v,
				   float *fn, float *gn,
				   float *p, float *p_0){
  /*  add_source(u,u0);
  add_source(v,v0);
  swap(u0,u); diffusion(1, u, u0, viscosity);
  swap(v0,v); diffusion(2, v, v0, viscosity);
  project(u,v,u0,v0);
  swap(u0,u); swap(v0,v);
  advection(1,u,u0,u0,v0);
  advection(2,v,v0,u0,v0);
  project(u,v,u0,v0);*/
  int i,j;
#pragma omp parallel for
  for(j = 0; j < M+2; j++){
    v[AT(1,j)] = 2*wall_velocity[0]-v[AT(0,j)];
    v[AT(N+1,j)] = 2*wall_velocity[2]-v[AT(N,j)];
  }
#pragma omp parallel for
  for(i = 0; i < N+2; i++){
    u[AT(i,1)] = 2*wall_velocity[3]-u[AT(i,0)];
    u[AT(i,M+1)] = 2*wall_velocity[1]-u[AT(i,M)];
  }
#pragma omp parallel for
  for(i = 1; i < N+1;i++){
#pragma omp parallel for
    for(j = 1; j<M+1;j++){
      fn[AT(i,j)] = u[AT(i,j)] + dt*(1/Re * (d2x(u,i,j) + d2y(u,i,j)) -
				     du2dx(u,i,j) - duvdy(u,v,i,j)+
				     Gx[AT(i,j)]);
      gn[AT(i,j)] = v[AT(i,j)] + dt*(1/Re * (d2x(v,i,j) + d2y(v,i,j)) -
				     duvdx(u,v,i,j) - dv2dy(v,i,j) +
				     Gy[AT(i,j)]);
    }
  }
  
#pragma omp parallel for
  for(j = 0;  j < M+2; j++){
    fn[AT(0,j)] = u[AT(0,j)];
    fn[AT(N+1,j)] = u[AT(N+1,j)];
  } 
#pragma omp parallel for
  for(i = 0; i < N+2; i++){
    gn[AT(i,0)] = v[AT(i,0)];
    gn[AT(i,M+1)] = v[AT(i,M+1)];
  }
#pragma omp parallel for  
  for(i = 1; i < N+1; i++){
#pragma omp parallel for
    for(j = 1; j < M+1; j++){
      RHS[AT(i,j)] = 1/dt *((fn[AT(i,j)]-fn[AT(i-1,j)])/dx +
			    (gn[AT(i,j)]-gn[AT(i,j-1)])/dy);
    }
  }
  
  int it = 0;
  
  while(it < max_iters){// && r2 > TOL){
#pragma omp parallel for
    for(j = 0; j <M+2;j++){
      p[AT(0,j)] = p_0[AT(1,j)];
      p[AT(N+1,j)] = p_0[AT(N,j)];
    }
#pragma omp parallel for
    for(i=0; i <N+2; i++){
      p[AT(i,0)] = p_0[AT(i,1)];
      p[AT(i,N+1)] = p_0[AT(i,N)];
    }
#pragma omp parallel for
    for(i = 1; i < N+1; i++){
#pragma omp parallel for
      for(j=1; j< M+1; j++){
	p[AT(i,j)] = (1-w)*p_0[AT(i,j)]+ w/(
				    (eE(i)+eW(i))/pow(dx,2) +
				    (eN(j)+eS(j))/pow(dy,2)) *
	  ((eE(i)*p_0[AT(i+1,j)] + eW(i)*p[AT(i-1,j)])/pow(dx,2) +
	   (eN(j)*p_0[AT(i,j+1)]+eS(j)*p[AT(i,j-1)])/pow(dy,2) - RHS[AT(i,j)]);
	/*	
	r[AT(i,j)] = (eE(i)*(p_0[AT(i+1,j)] - p_0[AT(i,j)]) -
		      eW(i)*(p_0[AT(i,j)] - p_0[AT(i-1,j)]))/pow(dx,2) +
	  (eN(j)*(p_0[AT(i,j+1)] - p_0[AT(i,j)]) -
	   eS(j)*(p_0[AT(i,j)] - p_0[AT(i,j-1)]))/pow(dy,2) - RHS[AT(i,j)];
	*/
      }
    }
    it++;
  }

  float u_max = -10000.0, v_max= -10000.0;
#pragma omp parallel for
  for(i = 1; i <N+1; i++){
#pragma omp parallel for
    for(j = 1; j <M+1; j++){
      u[AT(i,j)] = fn[AT(i,j)] - dt/dx *(p[AT(i+1,j)] - p[AT(i,j)]);
      v[AT(i,j)] = gn[AT(i,j)] - dt/dy *(p[AT(i,j+1)] - p[AT(i,j)]);
      if(u[AT(i,j)]>u_max) u_max = u[AT(i,j)];
      if(v[AT(i,j)]>v_max) v_max = v[AT(i,j)];

    }
  }
  float a = (Re/2)/(1/pow(dx,2) + 1/pow(dy,2));
  float b = dx/fabs(u_max);
  float c = dy;fabs(v_max);
  dt = tau *std::min(std::min(a,b),c);
  
}

void FluidSimulator::set_sources(float *d, float* u, float *v){
  d[AT(1,8)] = 10.0;
  //  for(int i = 0; i < N+1; i++){
  //    u[AT(i,M-2)] = 10;
  //  }
      //  u[AT(,50)] = 1;
  //  v[AT(50,50)] = 0.0;
}

void FluidSimulator::simulate(){
  set_sources(D_0, U_0, V_0);
  step_velocity(U, V, Fn, Gn, P, P_0);
  step_density(D, D_0, U, V, diffusion_rate);
}

