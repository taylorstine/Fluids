#include <iostream>
#include <stdlib.h>
#define FOR_EACH_CELL for(i=1; i<=N; i++){ for(j=1; j<=N; j++){
#define END_FOR }}
#define AT(i,j) ((i) + (N+2)*(j))
#define CUBE_SIZE 5

template<class T>
inline void SWAP(T& a, T& b){T c = a; a=b; b=c;}

enum boundary{DENSITY_BND, U_BND, V_BND};
enum wall{TOP, BOTTOM, LEFT, RIGHT};
enum boundary_type{NO_SLIP, FREE_SLIP, INFLOW, OUTFLOW};
static int boundaries[4];
static float wall_velocity[4];
static float inlet_velocity[4];
static bool particles;

void add_source(int N, float *x, float *s, float dt){
  int i, size = (N+2)*(N+2);
  for(i=0; i<size; i++) x[i] += dt*s[i];
}

void set_bnd(int N, int b, float *x, int *bnd){
  int i;
//#pragma omp parallel for
  for(i = 1; i <= N; i++){
    switch(b){
    case U_BND:
      switch(boundaries[LEFT]){
      case NO_SLIP: x[AT(0,i)] = -x[AT(1,i)]; break;
      case OUTFLOW: x[AT(0, i)] = x[AT(1,i)]; break;
      }
      switch(boundaries[RIGHT]){
      case NO_SLIP: x[AT(N+1,i)] = -x[AT(N,i)];; break;
      case INFLOW: x[AT(N+1,i)] = -inlet_velocity[RIGHT] - x[AT(N,i)]; break;
      case OUTFLOW: x[AT(N+1, i)] = x[AT(N,i)]; break;

      }
      switch(boundaries[BOTTOM]){
      case NO_SLIP: x[AT(i,1)] = wall_velocity[BOTTOM]; break;
      case OUTFLOW: x[AT(i, 1)] = x[AT(i,2)]; break;

      }
      switch(boundaries[TOP]){
      case NO_SLIP: x[AT(i,N)] = wall_velocity[TOP]; break;
      case OUTFLOW: x[AT(i,N)] = x[AT(i,N-1)]; break;
      }
      break;

    case V_BND:
      switch(boundaries[LEFT]){
      case NO_SLIP: x[AT(0,i)] = wall_velocity[LEFT]; break;
      case OUTFLOW: x[AT(0,i)] = x[AT(1,i)]; break;
      }
      switch(boundaries[RIGHT]){
      case NO_SLIP: x[AT(N+1,i)] = wall_velocity[RIGHT]; break;//x[AT(N,i)]; break;
      case INFLOW: x[AT(N+1,i)] = 0.0; break;
      case OUTFLOW: x[AT(N+1,i)] = x[AT(N,i)];
      }
      switch(boundaries[BOTTOM]){
      case NO_SLIP: x[AT(i,0)] = -x[AT(i,1)]; break;
      }
      switch(boundaries[TOP]){
      case NO_SLIP: x[AT(i,N+1)] = -x[AT(i,N)]; break;
      }
      break;
    case 0:
      x[AT(0,i)] =  x[AT(1,i)];
      x[AT(N+1,i)] =x[AT(N,i)];
      x[AT(i,0)] =  x[AT(i,1)];
      x[AT(i,N+1)] =x[AT(i,N)];
      break;
    }

    //    x[AT(0,i)]   = b==1 ? -x[AT(1,i)] : x[AT(1,i)];
    //    x[AT(N+1,i)] = b==1 ? -x[AT(N,i)] : x[AT(N,i)];
    //    x[AT(i,0)]   = b==2 ? -x[AT(i,1)] : x[AT(i,1)];
    //    x[AT(i,N+1)] = b==2 ? -x[AT(i,N)] : x[AT(i,N)];

  }
  x[AT(0,0)]    = 0.5f * (x[AT(1, 0 )] + x[AT( 0 ,1)]);
  x[AT(0, N+1)] = 0.5f * (x[AT(1,N+1)] + x[AT( 0 ,N)]);
  x[AT(N+1,0 )] = 0.5f * (x[AT(N, 0 )] + x[AT(N+1,1)]);
  x[AT(N+1,N+1)]= 0.5f * (x[AT(N, N+1)] + x[AT(N+1, N)]);

  int j;

//#pragma omp parallel for
  FOR_EACH_CELL
    if(bnd[AT(i,j)] == 1){
      for(int idx = 0; idx <= CUBE_SIZE; idx++){
	switch(b){
	case U_BND:
	  x[AT(i,j+idx)] = -x[AT(i-1,j+idx)];
	  x[AT(i+CUBE_SIZE, j+idx)] = -x[AT(i+CUBE_SIZE+1, j+idx)];
	  x[AT(i+idx,j)] = 0.0;
	  x[AT(i+idx,j+CUBE_SIZE)] = 0.0;
	  break;
	case V_BND:
	  x[AT(i,j+idx)] = 0.0;
	  x[AT(i+CUBE_SIZE,j+idx)] = 0.0;
	  x[AT(i+idx,j)] = -x[AT(i+idx, j-1)];
	  x[AT(i+idx,j+CUBE_SIZE)] = -x[AT(i+idx, j+CUBE_SIZE+1)];
	  break;
	default:
	  x[AT(i,j+idx)] = 0.0;//x[AT(i-1,j+idx)];
	  x[AT(i+CUBE_SIZE,j+idx)] = 0.0;//x[AT(i+CUBE_SIZE+1,j+idx)];
	  x[AT(i+idx,j)] = 0.0;//x[AT(i+idx, j-1)];
	  x[AT(i+idx,j+CUBE_SIZE)] = 0.0;//x[AT(i+idx, j+CUBE_SIZE+1)];
	  break;

	}
	//	x[AT(i,j-1)] = b == 0 ? x[AT(i,j)] : -x[AT(i,j)];
	//	x[AT(i,j+1)] = b == 0 ? x[AT(i,j)] : -x[AT(i,j)];
	//	x[AT(i-1,j)] = b == 0 ? x[AT(i,j)] : 0.0;
	//	x[AT(i+1,j)] = b == 0 ? x[AT(i,j)] : 0.0;

      }
    }
  END_FOR
  
 }

 void linear_solver(int N, int b, float *x,
		    float *x0, float a, float c, int *bnd){
   int i, j, k;
   for(k =0; k< 20; k++){
//#pragma omp parallel for
     FOR_EACH_CELL
       x[AT(i,j)] = (x0[AT(i,j)] + a *(x[AT(i-1, j)] + x[AT(i+1,j)]+
				       x[AT(i, j-1)] + x[AT(i,j+1)] ))/c;
     END_FOR
       set_bnd(N,b,x, bnd);
   }
 }

 void diffuse(int N, int b, float *x, float *x0,
	      float diff, float dt, int * bnd){
   float a = dt*diff*N*N;
   linear_solver(N, b, x, x0, a, 1+4*a, bnd);
 }

void advect_particles(int N, int b, float * d, float *d0,
	     float *u, float *v, float dt, int * bnd){
  int i, j, in, jn, i1, j1, i0, j0;
  float x, y, s0, t0, s1, t1, dt0;
  dt0 = dt*N;
//#pragma omp parallel for
  FOR_EACH_CELL
    x = i + dt0*u[AT(i,j)]; y = j + dt0*v[AT(i,j)];
  if(x<0.5f) x=0.5f; if(x>N+0.5f) x=N+0.5f; in=(int)x; i0 = (int)in+1;
  if(y<0.5f) y=0.5f; if(y>N+0.5f) y=N+0.5f; jn=(int)y; j0 = (int)jn+1;
  if(d[AT(in,jn)] == 0.0){
    d[AT(in,jn)] = d0[AT(i,j)];
    d[AT(i0,j0)] = 0.0;
  }
  else{
    d[AT(i,j)] = d0[AT(i,j)];
  }
  END_FOR
    set_bnd(N, b, d, bnd);


}
 void advect(int N, int b, float * d, float *d0,
	     float *u, float *v, float dt, int * bnd){

  int i, j, i0, j0, i1, j1;
  float x, y, s0, t0, s1, t1, dt0;
  dt0 = dt*N;
//#pragma omp parallel for
  FOR_EACH_CELL
    x= i - dt0*u[AT(i,j)]; y= j - dt0*v[AT(i,j)];
  if(x<0.5f) x=0.5f; if(x>N+0.5f) x=N+0.5f; i0=(int)x; i1 = i0+1;
  if(y<0.5f) y=0.5f; if(y>N+0.5f) y=N+0.5f; j0=(int)y; j1 = j0+1;
  s1 = x-i0; s0 = 1-s1; t1=y-j0; t0=1-t1;
  d[AT(i,j)] = s0*(t0*d0[AT(i0,j0)] + t1*d0[AT(i0,j1)]) + 
    s1*(t0*d0[AT(i1,j0)] + t1*d0[AT(i1,j1)]);
  END_FOR
    set_bnd(N, b, d, bnd);
}

void project(int N, float * u, float * v, float *p,
	     float * div, int *bnd){
  int i, j;
//#pragma omp parallel for
  FOR_EACH_CELL
  div[AT(i,j)] = -0.5f *(u[AT(i+1,j)] - u[AT(i-1,j)] +
			   v[AT(i,j+1)] - v[AT(i,j-1)])/N;
  p[AT(i,j)] = 0;
  END_FOR
    set_bnd(N, DENSITY_BND, div, bnd); set_bnd(N, DENSITY_BND, p, bnd);
  linear_solver(N, 0, p, div, 1, 4, bnd);

  //#pragma omp parallel for
  FOR_EACH_CELL
    u[AT(i,j)] -= 0.5f*N*(p[AT(i+1,j)] - p[AT(i-1,j)]);
  v[AT(i,j)] -= 0.5f*N*(p[AT(i,j+1)] - p[AT(i,j-1)]);
  END_FOR
    set_bnd(N, U_BND, u, bnd); set_bnd(N, V_BND, v, bnd);
}

void step_density(int N, float *x, float *x0,
		  float *u, float *v, float diff, float dt, int *bnd){
  add_source(N, x, x0, dt);
  SWAP(x0, x); diffuse(N, DENSITY_BND, x, x0, diff, dt, bnd);

  if(particles){
  SWAP(x0, x); advect_particles(N, DENSITY_BND, x, x0, u, v, dt, bnd);
  }else{
  SWAP(x0, x); advect(N, DENSITY_BND, x, x0, u, v, dt, bnd);
  }
}

void step_velocity(int N, float *u, float *v,
		   float *u0, float *v0, float visc, float dt, int *bnd){
  add_source(N, u, u0, dt); add_source(N, v, v0, dt);
  SWAP(u0, u); diffuse(N, 1, u, u0, visc, dt, bnd);
  SWAP(v0, v); diffuse(N, 2, v, v0, visc, dt, bnd);
  project(N, u, v, u0, v0, bnd);
  SWAP(u0, u); SWAP(v0, v);
  advect(N, 1, u, u0, u0, v0, dt, bnd); advect(N, 2, v, v0, u0, v0, dt, bnd);
  project(N, u, v, u0, v0, bnd);
}
