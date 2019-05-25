#include "GNUPlotter.h"
//#include <functional>
//#include <algorithm>    // for min
using namespace std;

//-------------------------------------------------------------------------------------------------
// convenience functions for certain types of plots (eventually move to class GNUPlotter):

template<class T>
void plotParametricSurface(
  const function<T(T, T)>& fx,
  const function<T(T, T)>& fy,
  const function<T(T, T)>& fz,
  int Nu, T uMin, T uMax,
  int Nv, T vMin, T vMax)
{
  // Create the data vector. The outer index runs over the indices for parameter u, the middle 
  // index runs over v and the innermost vector index runs from 0...2 giving a 3-vector containing 
  // x, y, z coordinates for each point:
  vector<vector<vector<double>>> d;              // doubly nested vector of data
  d.resize(Nu);                                  // we have Nu blocks of data
  for(int i = 0; i < Nu; i++) {                  // loop over the data blocks
    d[i].resize(Nv);                             // each block has Nv lines/datapoints
    T u = uMin + (uMax-uMin) * T(i) / T(Nu-1);   // value of parameter u
    for(int j = 0; j < Nv; j++) {                // loop over lines in current block
      T v = vMin + (vMax-vMin) * T(j) / T(Nv-1); // value of parameter v
      d[i][j].resize(3);                         // each datapoint has 3 columns/dimensions
      d[i][j][0] = fx(u,v);                      // x = fx(u,v)
      d[i][j][1] = fy(u,v);                      // y = fy(u,v)
      d[i][j][2] = fz(u,v);                      // z = fz(u,v)
    }
  }
  // maybe factor out a function addDataParametricSurface and have maybe have also functions: 
  // addDataParametricCurve2D, addDataParametricCurve3D, 

  // plot:
  GNUPlotter p;                                  // create plotter object
  p.addData(d);                                  // pass the data to the plotter         
  p.addCommand("set hidden3d");                  // don't draw hidden lines
  //p.addCommand("set view 20,50");                // set up perspective
  //p.addCommand("set lmargin 0");                 // margin between plot and left border
  //p.addCommand("set tmargin 0");                 // margin between plot and top border
  //p.addCommand("set ztics 0.5");                 // density of z-axis tics
  p.plot3D();                                    // invoke GNUPlot
}


template<class T>
void plotComplexSurface(const function<complex<T>(complex<T>)>& f,
  int Nr, T rMin, T rMax, int Ni, T iMin, T iMax)
{
  std::function<T(T, T)> fx, fy, fz;
  fx = [&] (T re, T im) { return real(f(complex<T>(re, im))); };
  fy = [&] (T re, T im) { return imag(f(complex<T>(re, im))); };

  // maybe let the user select, what should be used by a parameter:
  //fz = [&] (T re, T im) { return 0; };  // preliminary

  fz = [&] (T re, T im) { return re; };  // preliminary
  // for f(z) = z^2, this looks a lot like the Riemannian surface here:
  // https://www.youtube.com/watch?v=4MmSZrAlqKc&list=PLiaHhY2iBX9g6KIvZ_703G3KJXapKkNaF&index=13
  // ..but is it really the same thing?

  //fz = [&] (T re, T im) { return abs(f(complex<T>(re, im))); };
  //fz = [&] (T re, T im) { return abs(complex<T>(re, im)); };

  //fz = [&] (T re, T im) { return im; };  // preliminary


  plotParametricSurface(fx, fy, fz, Nr, rMin, rMax, Ni, iMin, iMax);
}
// maybe try showing abs and arg instead of re and im, also try to use abs and arg as inputs
// ...there seem to be a lot of combinations that may be tried

template<class T>
void plotVectorField2D(
  const function<T(T, T)>& fx,
  const function<T(T, T)>& fy,
  int Nx, T xMin, T xMax,
  int Ny, T yMin, T yMax)
{
  GNUPlotter::plotVectorField2D(fx, fy, Nx, xMin, xMax, Ny, yMin, yMax);
  // function now actually obsolete - we have all the code in class GNUPlotter now

  //p.addDataArrays(Nv, &x[0], &y[0], &dx[0], &dy[0], &c[0]);

  //p.addGraph("index 0 using 1:2:3:4:5 with vectors head size 0.2,10,30 filled lc palette notitle");
  //p.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette notitle");
  //p.addCommand("set palette gray negative");
  //p.addCommand("set palette rgbformulae 30,31,32 negative");

  //p.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette gray notitle");

  //p.plot();
  // -maybe give the user the option to scale the arrow-lengths
}
// info for drawing vector fields:
// https://stackoverflow.com/questions/5442401/vector-field-using-gnuplot
// http://www.gnuplotting.org/vector-field-from-data-file/
// for styling the arrows, see here:
// http://www.gnuplot.info/demo/arrowstyle.html
// -maybe make it possible to draw curves (i.e. integration paths) on top of the vector fields
// -how about equipotential lines?
// -can we similarly draw a vector field in 3D?

//-------------------------------------------------------------------------------------------------
// actual experiments:

void surfaceExperiment()
{
  // We plot a surface that is defined by 3 std::function objects for x(u,v), y(u,v), z(u,v)

  // Set up range and umber of sampling points for the two paremeters u and v for our parameteric 
  // surface:
  double uMin = -1; double uMax = +1; int Nu = 21;
  double vMin = -1; double vMax = +1; int Nv = 21; 

  // Define the 3 bivariate component functions as anonymous (lamda) functions assigned to 
  // std::function objects:
  std::function<double(double, double)> fx, fy, fz;
  fx = [] (double u, double v) { return u*v; };
  fy = [] (double u, double v) { return u+v; };
  fz = [] (double u, double v) { return u-v; };
  // maybe use a more interesting surface (torus, sphere, whatever)

  // plot the surface:
  plotParametricSurface(fx, fy, fz, Nu, uMin, uMax, Nv, vMin, vMax);
}

void complexExperiment()
{
  // Set up range and umber of sampling points for real and imaginary part:
  double rMin = -2; double rMax = +2; int Nr = 21;
  double iMin = -2; double iMax = +2; int Ni = 21;

  // Define the complex function w = f(z) = z^2 as example complex function:
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return z*z; };
  //f = [] (complex<double> z) { return z*z*z; };
  //f = [] (complex<double> z) { return z*z*z*z; };
  //f = [] (complex<double> z) { return exp(z); };
  //f = [] (complex<double> z) { return sin(2.0*z); };

  // plot the surface corresponding to the function:
  plotComplexSurface(f, Nr, rMin, rMax, Ni, iMin, iMax);

  // todo: try other ways to visualize a complex function - for example by showing, how grid-lines
  // are mapped (real, imag, radial, angular)
  // how about drawing curves in the z-plane (domain) and their image curves in the w-plane (range)
}


void vectorFieldExperiment()
{
  // Create two bivariate functions that resemble the complex function 
  // w = f(z) = z^2 = (x + iy)^2
  std::function<double(double, double)> fx, fy;
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }

  // plot the function as vector field:
  plotVectorField2D(fx, fy, 31, -3., +3., 21, -2., +2.);
}



/*
Ideas:
-figure out, how to plot 2D and 3D vector fields
-try to visualize complex functions as 2D vector fields





*/