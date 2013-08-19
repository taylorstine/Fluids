#include<iostream>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <SDL/SDL.h>
#include "math/math.hpp"
#include "mesh.hpp"
#include "fluid_simulator.hpp"

#ifdef OPENMP
#include <omp.h>
#endif

#define MAX_THREADS 20

#define SCREEN_WIDTH 700
#define SCREEN_HEIGHT 700
#define SCREEN_BPP 32
#define TRUE 1
#define FALSE 0

#define ASSERT(condition){if(!(condition)){std::cerr<<"ASSERTION FAILED: "<<#condition<<"@"<<__FILE__<<"("<<__LINE__<<")"<<std::endl;exit(EXIT_FAILURE);}}

SDL_Surface *surface;
int videoFlags;
bool Running;
bool Active;
Mesh mesh;
int Width = SCREEN_WIDTH;
int Height = SCREEN_HEIGHT;
FluidSimulator *simulation;
int N = 100;
int M = 100; 
float visc = 1e-10;
float diff = 100;
float dt = 0.05;
bool render_velocity = true;

struct Point{
  float x; float y;
};
struct Vector{
  Point p0; Point p1;
};

Vector *velocity_vectors;
GLuint *velocity_indices;



inline int AT(int i, int j){return i+(N+2)*j;}
template<class T>
inline void swap(T& a, T& b){T c = a; a=b; b=c;}

void Quit( int return_code){
  //  SDL_Quit();
  exit( return_code);
}

int initGL(void)
{
  glShadeModel(GL_SMOOTH);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  return (TRUE);
}

int resize_window(int width, int height)
{
  Width = width;
  Height = height;
  GLfloat ratio;
  if(height == 0)
    height = 1;
  ratio = (GLfloat)width/(GLfloat)height;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //  glOrtho(0.0, 1.0*ratio, 0.0, 1.0, -1.0, 1.0);
  glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  return(TRUE);
}


bool initSDL(){
  const SDL_VideoInfo * videoInfo;

  if(SDL_Init(SDL_INIT_VIDEO)<0){
    std::cout<<"error starting SDL  "<<SDL_GetError()<<std::endl;
    Quit(1);
  }
  videoFlags = SDL_OPENGL;
  videoFlags |= SDL_GL_DOUBLEBUFFER;
  videoFlags |= SDL_RESIZABLE;
  if(videoInfo->hw_available)
    videoFlags|= SDL_HWSURFACE;
  else
    videoFlags|= SDL_SWSURFACE;
  if(videoInfo->blit_hw)
    videoFlags|= SDL_HWACCEL;
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  surface = SDL_SetVideoMode(SCREEN_WIDTH, SCREEN_HEIGHT,
			     SCREEN_BPP, videoFlags);
  if(!surface){
    std::cout<<"Video mode set failed: "<<SDL_GetError()<<std::endl;
    Quit(1);
  }
  return true;
}

bool init_simulation(){
  simulation = new FluidSimulator(N, M);
  simulation->reset();
  return true;
}

int init()
{
  ASSERT(initSDL());
  ASSERT(initGL() == 1);
  ASSERT(create_mesh(mesh, N, M, SCREEN_WIDTH, SCREEN_HEIGHT));
  ASSERT(init_simulation());
  resize_window(SCREEN_WIDTH, SCREEN_HEIGHT);
#ifdef OPENMP
  omp_set_num_treads(MAX_THREADS);
#endif
  ASSERT(init_simulation());

  int size = (N+2)*(M+2);
  velocity_vectors = new Vector[size];
  velocity_indices = new GLuint [size*2];
  for(int i = 0; i < size*2; i++){
    velocity_indices[i] = i;
  }
  for(int i = 0; i < size; i++){
    velocity_vectors[i].p0.x = mesh.vertices[i].x;
    velocity_vectors[i].p0.y = mesh.vertices[i].y;
    velocity_vectors[i].p1.x = mesh.vertices[i].x;
    velocity_vectors[i].p1.y = mesh.vertices[i].y;
  }

  return (TRUE);
}

void handle_keypress(const SDL_keysym *keysym){
  switch(keysym->sym){
  case SDLK_ESCAPE:
    Running = false;
    break;
  case SDLK_F1: SDL_WM_ToggleFullScreen(surface); break;
  case SDLK_v: render_velocity = !render_velocity;
  case SDLK_r: simulation->reset();
  default: break;
  }
}

void handle_mouse_motion(SDL_MouseMotionEvent motion){
  //  std::cout<<motion.xrel<< " "<<motion.yrel<<" "<<motion.x<< " "<<motion.y<<std::endl;
  int x=0, y=0;
  if(SDL_GetRelativeMouseState(&x, &y) & SDL_BUTTON(1) && x!=0 && y!= 0){
    int i = floor(x * (N+2)/SCREEN_WIDTH);
    int j = floor(y * (M+2)/SCREEN_HEIGHT);
    simulation->add_force(motion.x, motion.y, i, j);
    std::cout<<"button 1 "<<x<<" "<<y<<std::endl;
  }
  if(SDL_GetRelativeMouseState(&x, &y) & SDL_BUTTON(2) && x!=0 && y!= 0){
    std::cout<<"button 2 "<<x<<" "<<y<<std::endl;
  }
  if(SDL_GetRelativeMouseState(&x, &y) & SDL_BUTTON(3) && x!=0 && y!= 0){
    std::cout<<"button 3 "<<x<<" "<<y<<std::endl;
  }

}

void handle_mouse_button(SDL_MouseButtonEvent button){
  int x, y;
  if(button.button == SDL_BUTTON_RIGHT){
    std::cout<<button.x<<" "<<button.y<<std::endl;
    x = button.x;
    y = Height-button.y;
    std::cout<<"x "<<x<<" "<<"y "<<y<<std::endl;
    int i = floor(x * (N+2)/Width);
    int j = floor(y * (M+2)/Height);
    simulation->add_density(i,j);
  }
  
}

void handle_event(){
  SDL_Event event;
  while(SDL_PollEvent(&event)){
    switch(event.type){
    case SDL_ACTIVEEVENT:
      //lost focus
      if(event.active.gain == 0)
	Active = true;
      else
	Active = true;
      break;
    case SDL_VIDEORESIZE:
      surface = SDL_SetVideoMode(event.resize.w,
				 event.resize.h,
				 SCREEN_BPP, videoFlags);
    case SDL_KEYDOWN: handle_keypress(&event.key.keysym); break;
    case SDL_MOUSEMOTION: handle_mouse_motion(event.motion); break;
    case SDL_MOUSEBUTTONDOWN: handle_mouse_button(event.button); break;
    case SDL_QUIT: Running = false; break;
    default: break;
    }
  }
}

void render_velocity_vectors(){
  glDisableClientState(GL_COLOR_ARRAY);
  glColor4f(1.0, 0.0, 0.0, 1.0);
  glDrawElements(GL_POINTS, mesh.num_vertices, GL_UNSIGNED_INT, mesh.point_indices);
  glVertexPointer(2, GL_FLOAT, 0, velocity_vectors);
  glDrawElements(GL_LINES, mesh.num_vertices*2, GL_UNSIGNED_INT, velocity_indices);
}

void render(){
  static GLint T0 = 0;
  static GLint Frames = 0;
  
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glColor4f(0.8, 0.0, 0.8, .5);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(2, GL_FLOAT, 0, mesh.vertices);
  glColorPointer(4, GL_FLOAT, 0, mesh.colors);
  glDrawElements(GL_TRIANGLES, mesh.num_triangles*3, GL_UNSIGNED_INT, &mesh.triangles[0]);

  glPointSize(1);
  glDisableClientState(GL_COLOR_ARRAY);
  glColor4f(1.0, 0.0, 0.0, 1.0);
  glDrawElements(GL_POINTS, mesh.num_vertices, GL_UNSIGNED_INT, mesh.point_indices);

  if(render_velocity)
    render_velocity_vectors();

    
  SDL_GL_SwapBuffers();
  Frames++;
  {
    GLint t = SDL_GetTicks();
    if(t - T0 >= 5000){
      GLfloat seconds = (t-T0)/1000.0;
      GLfloat fps = Frames/seconds;
      std::cout<<Frames<<" frames in "<<seconds<<" = "<<fps<<" FPS"<<std::endl;
      T0 = t;
      Frames = 0;
    }
  }

}

void update(){

  for(int i = 0; i < N+2; i++){
    for(int j = 0; j < M+2; j++){
      if(simulation->D_0[AT(i,j)] != simulation->D_0[AT(i,j)]){
	std::cout<<"error"<<std::endl;
      }
      ASSERT(simulation->D_0[AT(i,j)] == simulation->D_0[AT(i,j)]);
      mesh.colors[AT(i,j)].a = 1-simulation->D[AT(i,j)];
      velocity_vectors[AT(i,j)].p1.x = velocity_vectors[AT(i,j)].p0.x + simulation->U[AT(i,j)];
      velocity_vectors[AT(i,j)].p1.y = velocity_vectors[AT(i,j)].p0.y + simulation->V[AT(i,j)];
    }
  }
}

void main_loop(){
  Running = true;
  Active = true;
  while(Running){
    handle_event();
    if(Active){
      render();
      simulation->simulate();
      update();
    }
  }
}

void clean_up(){
  Quit(0);
}

int main(int argc, char **argv){
  init();
  main_loop();
  clean_up();
  return(0);
}
