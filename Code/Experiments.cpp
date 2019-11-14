#include "GNUPlotter.h"
#include "Experiments.h"
#include "MathTools.h"

//#include <math.h>
#include <random>

using namespace std;

#define M_PI 3.14159265358979323846

//std::vector<double>

double maxError(const std::vector<double>& x, const std::vector<double>& y)
{
  // todo: assert that x and y have the same size
  double maxErr = 0.0;
  for(size_t i = 0; i < x.size(); i++)
    if( fabs(x[i] - y[i]) > maxErr )
      maxErr = fabs(x[i] - y[i]);
  return maxErr;
}

std::vector<double> diff(const std::vector<double>& x)
{
  std::vector<double> d(x.size());
  for(int i = 0; i < x.size()-1; i++)
    d[i] = x[i+1] - x[i];
  d[d.size()-1] = 0;
  return d;
}

//-------------------------------------------------------------------------------------------------
// convenience functions for certain types of plots (eventually move to class GNUPlotter):

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

  GNUPlotter::plotSurface(fx, fy, fz, Nr, rMin, rMax, Ni, iMin, iMax);
}
// maybe try showing abs and arg instead of re and im, also try to use abs and arg as inputs
// ...there seem to be a lot of combinations that may be tried
// ...maybe this should be a class ComplexSurfacePlotter - for a function, the number of 
// parameters would be overwhelming

template<class T>
void plotComplexVectorField(const function<complex<T>(complex<T>)>& f,
  int Nr, T rMin, T rMax, int Ni, T iMin, T iMax)
{
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, false);
}
// maybe have a function plotComplexVectorFieldPolar? does that make sense? or maybe we should 
// generally allow for vector fields where the input is in polar coordinates? that may make sense
// in some cases, for example for a rotationally symmetric gravitational field
// -> plotVectorFieldPolar ...maybe factor out a function that creates the pairs of xy-values

// seems like sucha plot is called Polya-plot - see here
// http://mathworld.wolfram.com/PolyaPlot.html
// here at the bottom, they say something about Polya plots having an additional complex 
// conjugation involved - maybe let the complex conjugation be optional and on by default
// https://mathematica.stackexchange.com/questions/4244/visualizing-a-complex-vector-field-near-poles
// iirc, Visual Complex Analysis says soemthing about this - look up

// idea for visualizing a complex function:
// -if we pick a fixed radius r and let the angle p sweep through the range -pi..pi for the input 
//  z, we obtain a curve w = f(z) for that given p
// -if we draw these curves for a bunch of values of r, we get a 3-dimensional shape
// -we could use color to indicate the derivative (speed) of the curve traversal
// -of course, we could also use imaginary and real part instead of radius and angle but the latter
//  is attractive because the angle has a naturally finite range

template<class T>
void plotComplexPolarCurve(const function<complex<T>(complex<T>)>& f, T r, int N)
{
  std::vector<T> x(N), y(N);
  T p, dp = 2*M_PI/(N-1);
  complex<T> z, w;
  for(int i = 0; i < N; i++) {
    p = -M_PI + i * dp;
    z = std::polar(r, p);
    w = f(z);
    x[i] = w.real();
    y[i] = w.imag();
  }
  GNUPlotter::plot(N, &x[0], &y[0]);
}

// maybe draw a 3D curve and continuously increase r (input) and z (output coordinate)
template<class T>
void plotComplexCurve3D(const function<complex<T>(complex<T>)>& f,
  int N, T rMin, T rMax, T pMin, T pMax)
{
  //int N = Nr * Np;  // number of datapoints

  std::vector<T> x(N), y(N), z(N);
  T dp = (pMax-pMin)/(N-1);
  T dr = (rMax-rMin)/(N-1);
  complex<T> v, w;  // w = f(v) where v,w are complex numbers
  T r, p;
  for(int i = 0; i < N; i++) {
    p = pMin + i * dp;
    r = rMin + i * dr;
    v = std::polar(r, p);  // input to function
    w = f(v);              // output of function
    x[i] = w.real();
    y[i] = w.imag();
    z[i] = r;
  }

  //GNUPlotter::plot(N, &x[0], &y[0]);

  GNUPlotter plt;
  plt.addDataArrays(N, &x[0], &y[0], &z[0]);
  plt.plot3D();

  // todo: factor out a function that takes arrays for r and p instead of generating the grid on 
  // the fly inside the loop - allows custom grids

  // maybe indicate the phase of the input by the color of the line segment
}

// other idea: make a regular plot of the "landscape" of the absolute value but use hue as 
// indicator for the phase (brightness should be used by the renderer for lighting)
// ...maybe this is too sophisticated for gnuplot? maybe try in python with matplotlib or manim?

// see also here:
// https://www.johndcook.com/blog/2017/11/09/visualizing-complex-functions/
// https://www.amazon.de/Visual-Complex-Functions-Introduction-Portraits/dp/3034801793
// https://www.codeproject.com/Articles/80641/Visualizing-Complex-Functions
// https://www.wolfram.com/language/12/complex-visualization/
// https://www.pacifict.com/ComplexFunctions.html

// move to GNUPlotter:
template<class T>
void plotComplexFunctionReIm(const function<complex<T>(complex<T>)>& f,
  int Nr, T rMin, T rMax, int Ni, T iMin, T iMax)
{
  std::function<T(T, T)> fr, fi;
  fr = [&] (T re, T im) { return real(f(complex<T>(re, im))); };
  fi = [&] (T re, T im) { return imag(f(complex<T>(re, im))); };
  GNUPlotter plt;
  plt.addDataBivariateFunction(Nr, rMin, rMax, Ni, iMin, iMax, fr);
  plt.addDataBivariateFunction(Nr, rMin, rMax, Ni, iMin, iMax, fi);
  plt.plot3D();

  // maybe plot level lines
}

/*
template<class T>
void plotComplexArrayReIm(const T* x, const std::complex<T>* z, int N)
{
  std::vector<T> re(N), im(N);
  for(int i = 0; i < N; i++) {
    re[i] = z[i].real();
    im[i] = z[i].imag();
  }
  GNUPlotter plt;
  plt.addDataArrays(N, x, &re[0]);
  plt.addDataArrays(N, x, &im[0]);
  plt.plot();
}

template<class T>
void plotComplexArrayReIm(const std::complex<T>* z, int N)
{
  std::vector<T> x(N);
  GNUPlotter::rangeLinear(&x[0], N, T(0), T(N-1));
  plotComplexArrayReIm(&x[0], &z[0], N);
}
*/


//-------------------------------------------------------------------------------------------------
// actual experiments:

void curveExperiment2D()
{
  // we plot a Lissajous curve...
  double wx = 3, wy = 2;  // frequencies for x and y functions
  double ax = 1, ay = 1;  // amplitudes

  double tMin = 0; double tMax = 2*M_PI; int Nt = 200;

  std::function<double(double)> fx, fy;
  fx = [&] (double t) { return ax * cos(wx*t); };
  fy = [&] (double t) { return ay * sin(wy*t); };

  GNUPlotter::plotCurve2D(fx, fy, Nt, tMin, tMax);
}

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
  // maybe use a more interesting surface (torus, sphere, whatever), move to Demos, maybe let the 
  // user select the surface to be drawn - update the demo that currently creates the torus data
  // itself and uses low-level functions...but maybe, we should keep the demos for how to use the 
  // low-level functions...or point the user to look at the high-level functions, if they want to
  // replicate the drawings using lower level functions....hmmm...maybe have demoTorusLowLevel
  // and demoSurface

  // plot the surface:
  GNUPlotter::plotSurface(fx, fy, fz, Nu, uMin, uMax, Nv, vMin, vMax);
}

void complexExperiment() // rename
{
  // Set up range and umber of sampling points for real and imaginary part:
  double rMin = -3; double rMax = +3; int Nr = 31;
  double iMin = -3; double iMax = +3; int Ni = 31;

  // Define the complex function w = f(z) = z^2 as example complex function:
  function<complex<double>(complex<double>)> f;
  //f = [] (complex<double> z) { return z*z; };
  //f = [] (complex<double> z) { return z*z*z; };
  //f = [] (complex<double> z) { return z*z*z*z; };
  //f = [] (complex<double> z) { return 1./z; };         // 1st order pole at z=0

  f = [] (complex<double> z) { return z + 1./z; };
  // https://www.youtube.com/watch?v=rB83DpBJQsE
  // https://www.physik.uni-bielefeld.de/~borghini/Teaching/Hydrodynamics15/05_19.pdf
  // https://www.physik.uni-bielefeld.de/~borghini/Teaching/Hydrodynamics15/Hydrodynamics.pdf
  // -> page 61, eqIV.40: i think, we should use the negative derivative of 
  //    z + 1/z for the velocity field, not the function itself

  // https://en.wikipedia.org/wiki/Potential_flow_around_a_circular_cylinder
  // https://en.wikipedia.org/wiki/Potential_flow#Analysis_for_two-dimensional_flow

  //f = [] (complex<double> z) { return 1./(z+1.) + 1./(z-1.); };  // 2 poles at -1 and +1 (dipole..verify)
  //f = [] (complex<double> z) { return exp(z); };
  //f = [] (complex<double> z) { return sin(2.0*z); };

  // dipole: 1./(z-1.) + 1/(z+1.), maybe use 1./(z-1.) - 1/(z+1.) for source/sink (like positive 
  // and negative charge, ...but we need the inverse square law if we want to represent physical
  // situations...so our poles should be 2nd order
  // quadrupole: 1./(z-1.) + 1/(z+1.) + 1./(z-i) + 1/(z+i)

  // plot the surface corresponding to the function:
  //plotComplexSurface(f, Nr, rMin, rMax, Ni, iMin, iMax);
  //plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax);
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, true);
  GNUPlotter::plotComplexVectorField(f, Nr, rMin, rMax, Ni, iMin, iMax, false);

  // todo: try other ways to visualize a complex function - for example by showing, how grid-lines
  // are mapped (real, imag, radial, angular)
  // how about drawing curves in the z-plane (domain) and their image curves in the w-plane (range)
}

void complexCurve()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return z*z; };
  //f = [] (complex<double> z) { return z*z*z; };
  //f = [] (complex<double> z) { return 1./z; };
  //f = [] (complex<double> z) { return exp(z); };


  //plotComplexPolarCurve(f, 2.5, 100);
  plotComplexCurve3D(f, 1000, 0.0, 3.0, 0.0, 4*M_PI);
}

void complexReIm()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return z*z + 1.; };
  int N = 21;    // number of samples
  double r = 2;  // range from -r to +r (for both re and im)
  plotComplexFunctionReIm(f, N, -r, r, N, -r, r);
}


void vectorFieldExperiment()
{
  // Create two bivariate functions that resemble the complex function 
  // w = f(z) = z^2 = (x + iy)^2
  std::function<double(double, double)> fx, fy;
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }

  // todo: draw the vector field of a dipole, move to demos, maybe let the function have a 
  // parameter that selects, which particular function is drawn (z^2, dipole, quadrupole, etc.)

  // plot the function as vector field:
  GNUPlotter::plotVectorField2D(fx, fy, 31, -3., +3., 21, -2., +2.);



}
// info for drawing vector fields:
// https://stackoverflow.com/questions/5442401/vector-field-using-gnuplot
// http://www.gnuplotting.org/vector-field-from-data-file/
// for styling the arrows, see here:
// http://www.gnuplot.info/demo/arrowstyle.html

// -maybe give the user the option to scale the arrow-lengths


/** A class to wrap various functions related to visualizing the complex function f(z) = z^2.
..maybe have similar classes for ZedCubed, ZedInverse or OneOverZed, ExpZed, SineZed, etc. */

class ZedSquared
{

public:

  // x-component of 1st solution to the first order ODE system x' = x^2 - y^2, y' = 2*x*y which can 
  // be obtained by wolfram alpha via:
  // DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == 2 x[t] y[t]}, {x, y}, t] 
  // which gives two solutions
  static double fieldX1(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double x  = -(k2 * (t-2*C2)) / (1+4*k2*C2*C2 + k2*t*t - 4*k2*t*C2);
    return x;
    // x = -(e^(2 C1) (t - 2 C2))/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2);
  }

  static double fieldY1(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double a  = 1+4*k2*C2*C2 - 4*k2*t*C2 + k2*t*t;
    double b  = t - 2*C2;
    double s  = k2-(4*k2*k2*b*b) / a*a;
    double y  = (1./2) * (k1 + sqrt(max(0.0, s)));
    return y;
    // y = 1/2 (e^(C1) + sqrt(e^(2 C1) - (4 e^(4 C1) (t - 2 C2)^2)/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
  }

  static void fieldParamLimits1(double c1, double c2, double* tMin, double* tMax)
  {
    // Computes the limits between which the value under the square root in the above function
    // s = k^2-(4*k^4*b*b) / a*a is nonnegative - this is the range, which the parameter t is allowed
    // to traverse
    // sage:
    //  var("t a b k c2")
    //  a = 1+4*k^2*c2^2 - 4*k^2*t*c2 + k^2*t*t
    //  b = t - 2*c2;
    //  s = k^2-(4*k^2*k^2*b*b) / a*a;
    //  solve(s == 0, t)
    // gives:
    //  t == 1/2*(4*c2*k - 1)/k, t == 1/2*(4*c2*k + 1)/k

    double k = exp(c1);
    *tMin = (1./2)*(4*c2*k - 1)/k;   // ...simplify/optimize
    *tMax = (1./2)*(4*c2*k + 1)/k;
  }

  // 2nd solution:
  static double fieldX2(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double x  = -(k2 * (t+2*C2)) / (1+4*k2*C2*C2 + k2*t*t + 4*k2*t*C2);
    return x;
    // x = -(e^(2 C1) (t + 2 C2))/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  }

  static double fieldY2(double t, double C1, double C2)
  {
    double k1 = exp(C1);   // e^C1
    double k2 = k1*k1;     // e^(2 C1) = k1^2
    double a  = 1+4*k2*C2*C2 + 4*k2*t*C2 + k2*t*t;
    double b  = t + 2*C2;
    double s  = k2-(4*k2*k2*b*b) / a*a;
    double y  = (1./2) * (k1 - sqrt(max(0.0, s)));
    return y;
    // y = 1/2 (e^(C1) - sqrt(e^(2 C1) - (4 e^(4 C1) (t + 2 C2)^2)/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
  }
  // todo: refactor to avoid code duplication...maybe try to avoid recomputations - but to fully 
  // avoid them, we may need to be able to pass a vector-valued function to the plotter instead of 
  // two scalar valued functions (the computations for x and y have terms in common)

  // maybe try to use natural parametrization (nach bogenlänge parametrisieren) - find a function to
  // apply to t to find s...or actually we need t as function of s

  static void fieldParamLimits2(double c1, double c2, double* tMin, double* tMax)
  {
    // Computes the limits between which the value under the square root in the above function
    // s = k^2-(4*k^4*b*b) / a*a is nonnegative - this is the range, which the parameter t is allowed
    // to traverse
    // sage:
    //  var("t a b k c2")
    //  a = 1 + 4*k^2*c2^2 + 4*k^2*c2*t + k^2*t^2
    //  b = 2*c2 + t
    //  s = k^2-(4*k^2*k^2*b*b) / a*a;
    //  solve(s == 0, t)
    // gives:
    //  t == -1/2*(4*c2*k + 1)/k, t == -1/2*(4*c2*k - 1)/k

    double k = exp(c1);
    *tMin = (-1./2)*(4*c2*k + 1)/k;   // ...simplify/optimize
    *tMax = (-1./2)*(4*c2*k - 1)/k;
  }

};
void addZedSquaredFieldLine(GNUPlotter& plt, double c1, bool flipY = false) 
{
  // c1 controls the size of the loop that the field line draws, flipY lets the field line flipped
  // along the x-axis (i.e. the sign of y is flipped). somehow, we don't seem to get the lower 
  // halfplane field lines from the equations, so we cheat and just add the via symmetry - figure
  // out, why we don't get them naturally from the equations - have we missed a solution to the
  // ODE system?

  static int numFieldLines = 0; // quick and dirty counter - todo: make a class, use a member

  double c2 = 0;
  // c2 seems to have no visible effect (maybe it controls the speed and therefore the range 
  // tMin..tMax?)

  double sign = 1; if(flipY) sign = -1;

  std::function<double(double)> gx, gy;
  double tMin, tMax;
  std::string s2;
  std::string s1 = "index ";
  std::string s3 = " using 1:2 with lines lt 1 notitle"; 

  int numPoints = 401; 
  // maybe this should depend on the total length of the field line - evaluate the line integral
  // for length computation

  // add data and commands for the first half of the field line:
  s2 = to_string(2*numFieldLines+1);
  ZedSquared::fieldParamLimits1(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        ZedSquared::fieldX1(t, c1, c2); };
  gy = [&] (double t) { return sign * ZedSquared::fieldY1(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3); 

  // add data and commands for the second half of the field line:
  s2 = to_string(2*numFieldLines+2);
  ZedSquared::fieldParamLimits2(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        ZedSquared::fieldX2(t, c1, c2); };
  gy = [&] (double t) { return sign * ZedSquared::fieldY2(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3);

  numFieldLines++;
}
void zedSquaredVectorField()
{
  // We plot the 2D vector field corresponding to the complex function f(z) = z^2 and also draw
  // curves that represents a field lines

  // create and set up plotter:
  GNUPlotter plt;                 // plotter object
  plt.clearCommandFile();         // we don't want to use the default line styles/colors
  plt.setGrid();
  plt.setGraphColors("209050");   // field line color, visible on top of the arrows

  // add data for the 2 bivariate functions and commands for plotting the vector field:
  std::function<double(double, double)> fx, fy; // vector field fx(x,y), fy(x,y)
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }
  //plt.addDataVectorField2D(fx, fy, 15, -1.5, 1.5, 31, -3.0, +3.0);
  plt.addDataVectorField2D(fx, fy, 21, -2.0, 2.0, 31, -3.0, +3.0);
  plt.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette notitle");
  plt.addCommand("set palette rgbformulae 30,31,32 negative");

  // add data and commands for field lines:
  double cMin = -3.0, cMax = 1.0, cStep = 0.5;
  double c = cMin;
  while(c <= cMax) {
    addZedSquaredFieldLine(plt, c, false);
    addZedSquaredFieldLine(plt, c, true);
    c += cStep;
  }
  // obtaining the field lines for the bottom halfplane by reflection is actually cheating - i 
  // think, they should somehow naturally fall out of the equations - but for some reason didn't


  plt.plot();
  // todo:
  // -add arrows and maybe something that let's use see the speed (maybe plot segments where the 
  //  particle is fast fainter - resembles an analog oscilloscope look)
  // -maybe we could add tangent/velocity vectors on the field line..this would amount to just 
  //  evaluate the vector field at these points - but if all we have is the parametric equations,
  //  we could also use numerical derivatives
}
// -how about equipotential lines? for this, we perhaps first should figure out how to draw several
//  curves on top of a scalar field in general

// todo: plot vector field of a polynomial with zeros placed at -1, +1, -i, +i, 0, maybe
// also plot vector fields of rational functions

std::complex<double> rationalVectorField(std::complex<double> z)
{
  // Computes the value of a rational function
  complex<double> i(0, 1);  // imaginary unit
  return (z-1.) * (z+1.) * (z-i) * (z+i) / z;
  // maybe try more interesting patterns of poles and zeros, try visualizing filter transfer 
  // functions as vector fields
}
void demoVectorField()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return rationalVectorField(z); };
  GNUPlotter::plotComplexVectorField(f, 31, -1.5, +1.5, 31, -1.5, +1.5, false);
}



void testInitialValueSolver()
{
  // We test the initial value solver by throwing the simple differential equation y' = k*y at it
  // and compare the produced results with the analytic result given by y(t) = exp(k*t). We also 
  // compare the different accuracies obtained by different stepper methods (Euler, midpoint, etc.)

  double k = -1.0;  // constant in y' = k*y
  double h =  0.1;  // step size
  int N = 50;       // number of datapoints

  // create data-arrays and objects:
  double s[2];      // state
  std::function<void (const double *y, double *yp)> f;
  f = [&] (const double *y, double *yp) { 
    yp[0] = 1.0;
    yp[1] = k * y[1];
  };
  std::vector<double> tA(N), tE(N), tM(N), yA(N), yE(N), yM(N);
  InitialValueSolver<double> solver;
  solver.setDerivativeFunction(f, 2);
  solver.setStepSize(h);

  // produce analytic solution for reference:
  for(int n = 0; n < N; n++) {
    tA[n] = h*n;  yA[n] = exp(k*tA[n]); }

  // produce output via Euler steps:
  s[0] = 0.0; s[1] = 1.0;
  for(int n = 0; n < N; n++) { 
    tE[n] = s[0]; yE[n] = s[1]; solver.stepEuler(&s[0], &s[0]); }

  // produce output via midpoint method:
  s[0] = 0.0; s[1] = 1.0;
  for(int n = 0; n < N; n++) { 
    tM[n] = s[0]; yM[n] = s[1]; solver.stepMidpoint(&s[0], &s[0]); }

  // compute the maximum error for each method:
  double eE, eM;  
  eE = maxError(yA, yE);  // Euler error
  eM = maxError(yA, yM);  // midpoint error
  // this works only for fixed stepsize!
  // ..ok - yes - midpoint error is much less than Euler error, as it should be
  // todo: try generalized Euler based on non-Newtonian calculus - with this, the Euler method 
  // should produce exact results


  GNUPlotter plt;
  plt.addDataArrays(N, &tA[0], &yA[0]);
  plt.addDataArrays(N, &tE[0], &yE[0]);
  plt.addDataArrays(N, &tM[0], &yM[0]);
  plt.plot();
}







std::complex<double> complexDipoleField(std::complex<double> z,
  std::complex<double> cl = -1.0, std::complex<double> cr = +1.0)
{
  // Computes the field of a dipole with two charges at z = -1 and z = +1 with charge cl and cr
  // respectively (cl, cr stand for for left and right charge)
  // ...but not physically correct...

  typedef std::complex<double> Complex;

  Complex pl(-1.0, 0.0);     // position of left charge
  Complex pr(+1.0, 0.0);     // position of right charge
  double dl  = abs(z-pl);    // distance of z to left charge
  double dr  = abs(z-pr);    // distance of z to righ charge

  //double dl3 = dl*dl*dl;     // cubes of... 
  //double dr3 = dr*dr*dr;     // ...the distances
  //double dl2 = dl*dl;     // squares of... 
  //double dr2 = dr*dr;     // ...the distances
  // the physically correct law (using the cubes) doesn't look good (we need the cube because the
  // distance appears also in the numerator (hidden in the non-normalized vector))
  
  return cl*(z-pl)/dl + cr*(z-pr)/dr; // or does it have to be pl-z ?
}
void demoComplexDipole()
{
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return complexDipoleField(z); };
  GNUPlotter::plotComplexVectorField(f, 41, -2., +2.0, 21, -1.0, +1.0, false);
}
// maybe start with the equation for the potential, the field-lines should then go into the 
// direction of the gradient - maybe this can be done automatically by computing numeric gradients
// -> have a function addPotentialFieldLines2D

//
template<class T>
int findBin(T x, int numBinEdges, T* binEdges) // numBinEdges == numBins+1
{
  // x is out of range (to the right):
  if(x > binEdges[numBinEdges-1] )
    return numBinEdges;

  // return the bin index or -1, if x is out of range to the left:
  int i = -1;
  while( i <= numBinEdges-2 && x >= binEdges[i+1] )
    i++;
  return i;
}

bool testFindBin() // unit test for the findBin function
{
  int binEdges[5] = { 2,3,5,6,8 };
  int numBins  = 4;
  int numEdges = numBins+1;
  bool r = true;
  r &= findBin(1, numEdges, binEdges) == -1; // x too low
  r &= findBin(2, numEdges, binEdges) ==  0;
  r &= findBin(3, numEdges, binEdges) ==  1;
  r &= findBin(4, numEdges, binEdges) ==  1;
  r &= findBin(5, numEdges, binEdges) ==  2;
  r &= findBin(6, numEdges, binEdges) ==  3;
  r &= findBin(7, numEdges, binEdges) ==  3;
  r &= findBin(8, numEdges, binEdges) ==  4; 
  r &= findBin(9, numEdges, binEdges) == numEdges;// x too high
  return r;
}

template<class T>
void plotHistogram(int numDataPoints, T* data, int numBinEdges, T* binEdges, bool relative = true)
{
  int numBins = numBinEdges-1;
  std::vector<T> p(numBins); // "probability"

  for(int i = 0; i < numDataPoints; i++) {
    int bin = findBin(data[i], numBinEdges, binEdges);

    if(bin >= 0 && bin < numBins) 
      p[bin] += 1; // todo: allow for a weight, i.e. p[bin] += weights[i]
  }

  // optionally normalize:
  if(relative)
    for(int i = 0; i < numBins; i++)
      p[i] /= numDataPoints;  // or, in general, divide by sum-of-weights

  GNUPlotter plt;
  //plt.addDataArrays(numBins, x, p);
  //plt.addCommand("set boxwidth 0.75");
  //plt.addGraph("index 0 using 1:2 with boxes fs solid lc rgb \"#a00000ff\" notitle"); // 1st channel is alpha
  //plt.plot();
}
// move to GNUPlotter
// see here for inspiration for signature and features:
// https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.hist.html
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html#numpy.histogram




// matplotlib's behavior is:
// if bins is: [1, 2, 3, 4], then the first bin is [1, 2) (including 1, but excluding 2) and the 
// second [2, 3). The last bin, however, is [3, 4], which includes 4.
// ...it's probably a good idea to mimic this behavior

void testHistogram()
{
  testFindBin();

  const int N = 10000;  // number of experiments
  const int numBins = 20;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(10.0,2.0);


  double x[numBins];
  GNUPlotter::rangeLinear(x, numBins, 0.0, (double)(numBins-1));

  double p[numBins] = {};  // this holds the number of elements per bin

  for(int i = 0; i < N; i++) 
  {
    double number = distribution(generator);
    if((number >= 0.0) && (number < numBins)) {
      // figure out, into which bin this number belongs:
      int bin = int(number); 
      // maybe use rounding or more sophisticated ways for defining bins - maybe by min/max values
      // maybe have a function findBin(T* value, T* binLimits, T* numBins) that implements binary
      // search


      p[bin] += 1.0;
    }
  }

  // normalize p by dividing by N - plot the relative numbers of occurences
  for(int i = 0; i < numBins; i++)
    p[i] /= N;


  GNUPlotter plt;
  plt.setRange(0.0, 20.0, 0.0, 0.2);
  plt.addDataArrays(numBins, x, p);
  plt.addCommand("set boxwidth 0.75");
  //plt.addGraph("index 0 using 1:2 with boxes fs solid 0.50 notitle");
  plt.addGraph("index 0 using 1:2 with boxes fs solid lc rgb \"#a00000ff\" notitle"); // 1st channel is alpha
  plt.plot();
  // todo: 
  // -fill boxes with color (maybe semitransparent?)
  //  http://gnuplot.sourceforge.net/demo/fillstyle.html
  // -maybe use non-equidistant bin boundaries to 
  // make it more interesting (denser toward the center)....but then we may have to normalize the
  // heights by the widths ...or something - otherwise the narrower bins are too short
  // the boxes are centered at the given x-values - to plot multiple histograms in one plot, we
  // need to offset the boxes for each dataset


  // todo:
  // make a histogram with 3 datasets and plot them as red, green and blue bars - ues different
  // means and variances for each dataset
  //....maybe, for audio, such histograms can be used for raw sample amplitude values (use colors
  // for the channels) - useful for analyzing the behavior of noise generators - maybe have a 
  // function plotHistogram(N, T* data, int numBins, T minX, T maxX) ...and/or
  // plotMultiHistogram(T minX, T maxX, int numBins, int numDataPoints, T* data1, T* data2, ...)
  // and maybe have versions that let teh user define the bins-limits explicitly as arrays
 
  int dummy = 0;
}
// http://www.cplusplus.com/reference/random/normal_distribution/
// https://stackoverflow.com/questions/2471884/histogram-using-gnuplot
// http://gnuplot.sourceforge.net/demo/histograms.html
// http://gnuplot.sourceforge.net/docs_4.2/node249.html


void testMoebiusStrip()
{
  int N = 1; // number of half-turns, 1: classical Moebius, even: orientable, odd: non-orientable
  double L = 0.25;
  std::function<double (double u, double v)> fx, fy, fz;
  fx = [&](double t, double p) { return (1+t*cos(N*p/2))*cos(p); };
  fy = [&](double t, double p) { return (1+t*cos(N*p/2))*sin(p); };
  fz = [&](double t, double p) { return    t*sin(N*p/2); };

  GNUPlotter::plotSurface(fx, fy, fz, 5, -L, L, 50, 0., 2.*M_PI);
}
// todo: try to make a Moebius trefoil knot

template<class T>
void allocateMatrix(T**& A, int N, int M)
{
  A = new T*[N];
  for(int i = 0; i < N; i++)
    A[i] = new T[M];
}

template<class T>
void freeMatrix(T**& A, int N, int M)
{
  for(int i = 0; i < N; i++)
    delete[] A[i];
  delete[] A;
}

// for a positive modulus m, this returns the mathematically proper x mod m
int wrap(int x, int m)
{
  while(x <  0) x += m;
  while(x >= m) x -= m;
  return x;
}

void testSchroedinger()
{
  // doesn't work yet - the numerical solver explodes - maybe the equation is wrong or the solver
  // messes up - maybe try an example with the regular wave equation to figure out a working solver
  // scheme, then apply it to the schroedinger equation
  // i think, it's the numreical solver - there are growing osillations at both ends - i think, 
  // it's unstable ...hmm - it seems to depend on the seetings of mass, spatial and temporal 
  // oversampling, etc - the current settings seem to work.
  // maybe implement this stuff in a class rsQuantumParticle or rsQuantumOscillator in the 
  // QuantumSystems.h/cpp file in the rs-met codebase, functions:
  // setInitialWaveFunction, setPotentialFunction, setMass, setSpringConstant, 
  // setSpatialOversampling, setTemporalOversampling, setSpatialRange, 
  // setDefaultInitialWaveFunction(type, parameter), etc.
  
  // Observations:
  // -a free particle spreads out linearly in time - this is because there's uncertainty in 
  //  position *and* momentum (i.e. velocity). if at t = 0 it is known that the particle's position 
  //  x is in 2..3 and velocity is in 9..11, then at t = 1 the position is in 2+9..3+11 = 11..14 and 
  //  this new interval is wider - the initial width is 3-2 = 1 and the final with is 
  //  initial_width + 1*momemtum_width = (3-2) + 1*(11-9) = 1+2 = 3

  // https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation
  // https://en.wikipedia.org/wiki/Wave_packet
  // https://en.wikipedia.org/wiki/Free_particle
  // https://en.wikipedia.org/wiki/Particle_in_a_box - approximate the infinite potential well with a V(x) = x^(2*n) for large n
  // https://en.wikipedia.org/wiki/Particle_in_a_ring
  // https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator

  static const int numSpaceSamples = 81;
  static const int numTimeSamples  = 81;
  static const int timeOversample  = 100;  // time step should be smaller than space step
  static const int spaceOversample = 1;

  double xMax = 1.0;   // also let user speficy xMin
  double tMax = 1.0;
  double hBar = 1;
  double m    = 100;  // mass
  double k    = 300;  // spring constant - larger values hold the thing together more strongly, 
                      // 0 gives a free particle


  double w = sqrt(k/m); // angular frequency of oscillator

  // allocate arrays for plotting:
  double *t = new double[numTimeSamples];   // time axis for plot
  double *x = new double[numSpaceSamples];
  GNUPlotter::rangeLinear(t, numTimeSamples,  0.0, 1.0);
  GNUPlotter::rangeLinear(x, numSpaceSamples, 0.0, 1.0);
  double **zr; allocateMatrix(zr, numTimeSamples, numSpaceSamples); // real part
  double **zi; allocateMatrix(zi, numTimeSamples, numSpaceSamples);
  double **za; allocateMatrix(za, numTimeSamples, numSpaceSamples);

  // arrays for the (oversampled) computations:
  int Nt = numTimeSamples * timeOversample;
  int Nx = numSpaceSamples * spaceOversample;
  typedef std::complex<double> Complex;
  Complex** Psi;  allocateMatrix( Psi, Nt, Nx);    // wave-function (of space and time)
  Complex** Psi_t; allocateMatrix(Psi_t, Nt, Nx);  // time derivative of wave function
  std::vector<Complex> Psi_xx(Nx);                 // 2nd spatial derivative of Psi per iteration


  // define the potential function:
  std::function<Complex (double x)> V;
  //V = [&](double x) { return 0.0; };             // no potential -> free particle


  V = [&](double x) { 
    double xs = x-0.5; // shifted x - get rid - we just use it here becs our x-range is centered at 0.5
    return 0.5*m*w*w * xs*xs; 
  }; // quadratic potential -> harmonic oscillator
  // oh - but our our x-range is cneterd at 05 - shourd be at 0 -> define a range for x (min/max)


  // initialize wave-function - give it an initial shape in space of a gaussian bump in the center:
  int ti, xi;  // temporal and spatial loop indices
  Complex i(0,1);              // imaginary unit
  double mu = 0.5;
  double sigma = 0.05;
  for(int xi = 0; xi < Nx; xi++) {
    double x = double(xi) / (Nx-1);
    double gauss = exp(-(x-mu)*(x-mu) / (sigma*sigma));
    Psi[0][xi] = i * gauss;
  }
  // todo: 
  //  -maybe multiply by +i or -i or maybe exp(i*phi) for phi being an arbitrary angle
  //  -is this how we give it an initial velocity? nope! but how do we? i think, we need to shift 
  //   the Fourier trafe to a frequency other than 0 - by multiplying it with exp(i*lamda*x)? - we 
  //   probably should not multiply by a real sinusoid because that would give a symmetric 
  //   magnitude spectrum for the momentum
  // -maybe multiply by a Hermite polynomial this gives the eigenfunctions of the harmonic 
  //  oscillator - the order of the polynomial is the energy level 
//    https://en.wikipedia.org/wiki/Hermite_polynomials
  // maybe normalize to unit mean

  //GNUPlotter::plotComplexArrayReIm(Psi[0], Nx);

  // solve Schroedinger equation numerically by forward Euler method in time and central 
  // differences in space:

  //Complex k = (i*hBar)/(2*m);  // scaler for spatial derivative to get time deriavtive
  double dt = tMax / (Nt-1);   // temporaly sampling interval
  double dx = xMax / (Nx-1);   // spatial sampling interval
  for(ti = 1; ti < Nt; ti++)   // ti: time index
  {
    // compute second spatial derivative of wave function Psi by central differences (treating the
    // ends cyclically):
    for(xi = 0; xi < Nx; xi++)
      Psi_xx[xi] = (Psi[ti-1][wrap(xi-1,Nx)] + Psi[ti-1][wrap(xi+1,Nx)] - 2.*Psi[ti-1][xi])/(dx*dx);

    //// no wrap
    //for(xi = 1; xi < Nx-1; xi++)
    //  Psi_xx[xi] = (Psi[ti-1][wrap(xi-1,Nx)] + Psi[ti-1][wrap(xi+1,Nx)] - 2.*Psi[ti-1][xi])/(dx*dx);

    // compute time derivative and update wave function:
    for(xi = 0; xi < Nx; xi++) {
      //Psi_t[ti-1][xi] = k*Psi_xx[xi];        // Eq 9.4 in Susskind  

      double x = double(xi) / (Nx-1); 
      // the x-array can't be used bcs. it's not oversampled todo: include a minimum xMin that's
      // not necessarily zero

      // compute time derivative of the wave function via the Schroedinger equation:
      // Psi_t = (i*hBar)/(2*m)*Psi_xx - (i/hBar)*V*Psi:
      Psi_t[ti-1][xi] = ((i*hBar)/(2*m)) * Psi_xx[xi]     // term for free particle
                       -((i/hBar)* V(x)) * Psi[ti-1][xi]; // term from the potential

      // update the wave function:
      Psi[ti][xi]  = Psi[ti-1][xi] + dt * Psi_t[ti-1][xi];
    }
    //Psi[ti][0] = Psi[ti][1]; // test



    if(ti % (timeOversample/1) == 0)
    {
      //GNUPlotter::plotComplexArrayReIm(&Psi_xx[0], Nx);  // 2nd spatial derivative
      //GNUPlotter::plotComplexArrayReIm(dPsi[ti],   Nx);  // update
      //GNUPlotter::plotComplexArrayReIm(Psi[ti],    Nx);
    }
  }

  GNUPlotter::plotComplexArrayReIm(Psi[Nt-1], Nx);


  // from the oversampled computation result, obtain the result for plotting by downsampling:
  for(ti = 0; ti < numTimeSamples; ti++) {
    for(xi = 0; xi < numSpaceSamples; xi++) {
      zr[ti][xi] = Psi[ti*timeOversample][xi*spaceOversample].real();
      zi[ti][xi] = Psi[ti*timeOversample][xi*spaceOversample].imag(); 
      za[ti][xi] = abs(Psi[ti*timeOversample][xi*spaceOversample]);}}

  // plot:
  GNUPlotter plt;
  //plt.addCommand("set view 50,260"); // todo: add member function setView to GNUPlotter
  //plt.plotSurface(numTimeSamples, numSpaceSamples, t, x, zr);
  //plt.plotSurface(numTimeSamples, numSpaceSamples, t, x, zi);
  //plt.plotSurface(numTimeSamples, numSpaceSamples, t, x, za);

  
  plt.addDataMatrix(numTimeSamples, numSpaceSamples, t, x, za);
  plt.setPixelSize(450, 400);
  plt.addCommand("set size square");                      // set aspect ratio to 1:1
  plt.addGraph("i 0 nonuniform matrix w image notitle");   
  //plt.addCommand("set palette color");                  // this is used by default
  //plt.addCommand("set palette color negative");         // reversed colors
  plt.addCommand("set palette gray");                   // maximum is white
  //plt.addCommand("set palette gray negative");          // maximum is black
  //plt.addCommand("set palette rgbformulae 30,31,32");     // colors printable as grayscale
  plt.plot();
  


  // clean up:
  delete[] t;
  delete[] x;
  freeMatrix(zr, numTimeSamples, numSpaceSamples);
  freeMatrix(zi, numTimeSamples, numSpaceSamples);
  freeMatrix(za, numTimeSamples, numSpaceSamples);
  freeMatrix(Psi,   Nt, Nx);
  freeMatrix(Psi_t, Nt, Nx);
}

// move to GNUPlotter:
void showMultiPlot(GNUPlotter& p, int numRows, int numCols)
{
  bool top = true;  // y coordinate starts at top and goes downward (rename, make parameter)
  double h = 1.0 / numRows;                         // relative height of subplots
  double w = 1.0 / numCols;                         // relative width of subplots
  p.addCommand("set multiplot");                    // init multiplot
  p.addCommand("set size "+p.s(w)+","+p.s(h)+"\n"); // set size of subplots
  std::string str;                                  // for accumulating command strings
  for(int i = 0; i < numRows; i++) {                // loop over the plot-rows
    for(int j = 0; j < numCols; j++) {              // loop over the plot-columns

      // set subplot position:
      double x0 = j*w;
      double y0;
      if(top) y0 = 1.0 - i*h - h;
      else    y0 = i*h;
      str  = "set origin ";
      str += p.s(x0) + ",";
      str += p.s(y0) + "\n";
      p.addCommand(str);

      // add subplot command:
      int index = numCols*i + j;                    // index of the dataset
      str = "plot '" + p.getDataPath() + "' i ";
      str += p.s((unsigned int)index);
      str += " u 1:2";
      str += " w lines lw 1.5 notitle";             // make it a style-parameter string
      p.addCommand(str);
    }
  }
  p.addCommand("unset multiplot");
  p.invokeGNUPlot();
}

void testMultiPlot()  // or maybe we should call it 3x4?
{
  // Creates a multiplot grid with lissajous figures.

  // Settings:
  int N       = 201;        // number of datapoints per plot
  int numRows = 5;          // number of rows
  int numCols = 5;          // number of columns
  int size    = 120;        // number of pixels per subplot

  // Generate data and add it to the datafile (this is not the most economic way to do it):
  GNUPlotter p;                             // create a plotter object
  std::vector<double> x(N), y(N);           // allocate memory for data
  for(int i = 1; i <= numRows; i++) {       // loop over the plot-rows
    for(int j = 1; j <= numCols; j++) {     // loop over the plot-columns
      for(int n = 0; n < N; n++) {          // loop over datapoints for current plot
        double t = n*2*M_PI / (N-1);        // compute curve parameter
        x[n] = sin(i*t);                    // compute x-coordinate
        y[n] = sin(j*t); }                  // compute y-coordinate
      p.addDataArrays(N, &x[0], &y[0]); }}  // add dataset to file

  // add the subplot commands to the commandfile and plot:
  p.setPixelSize(numCols*size, numRows*size);
  p.addCommand("set size square");      // apsect ratio of subplots 1:1
  p.addCommand("unset xtics"); 
  p.addCommand("unset ytics");

  // what unit is this? inches?
  p.addCommand("set lmargin 0.2");
  p.addCommand("set rmargin 0.2");
  p.addCommand("set tmargin 0.2");
  p.addCommand("set bmargin 0.2");

  showMultiPlot(p, numRows, numCols);
}
// maybe try to plot a set of modes for a circular membrane - i.e. a multiplot of polar 3D plots

//  -see 
//  -factor out: 
//    -initMultiPlot, showMultiPlot
//    -addSubPlot(double x, double y, double w, double h, ...)
//     ..this should reverse the y-axis, i.e. start from top...or maybe see how 
//     matplotlib handles this
// maybe the plot command can take a title parameter

// todo: try to let




/*

Ideas:
-plot a bunch of field-lines - maybe use as example 2 charges at (-1,0) and (+1,0), start the 
 field-lines in the middle vertical line (0, y) and from there, integrate them numerically into
 both directions...maybe using a function addBiDirectionalFieldLine
-make a demo showing a positive charge at +1 and a negative charge at -1 using the physically 
 correct law for the electric field - make another plot with two negative and two positive charges
 the two negative charges should look like 2 gravitational fields - call it demoDipole

-plot complex mapping - maybe there are various ways to do this...


-plot 3D curves (trefoil knot)
-plot 3D vector fields
-plot curves on top of a 2D vector field - can be used to show equipotential curves and/or 
 integration paths

 weitz on visualization:
 https://www.youtube.com/watch?v=BhtnlKOC-0s&t=189s
 https://www.youtube.com/watch?v=BhtnlKOC-0s


Field lines:
-draw field lines of 2D vector fields - how do we find them? start somewhere and follow the field?
 ...but what about error accumulation? ...in any case, it seems we need a general way to add many
 curves to a vector field (for field lines of vector fields and equipotential lines for scalar
 fields)
 https://www.quora.com/How-can-I-find-an-expression-for-vector-field-lines
 https://math.stackexchange.com/questions/1992208/how-to-find-the-field-lines-of-a-vector-field
-maybe we should plot the field lines as parametric curves whose equations should be obtained
 analytically? maybe like this:
 -let f(x,y) and g(x,y) be the functions that define our vector field
 -we want a family of parametric curves x(t), y(t) such that in any point x,y on one of the curves, 
  the curve's tangent is in the direction of the vector field - so we require: 
  x'(t) = f(x(t),y(t)), y'(t) = g(x(t),y(t)) for any t, or shorter: x' = f(x,y), y' = g(x,y)
  -> this is a system of 2 first order differential equations
  -> maybe if f and g are simple enough, we can find an analytic solution to this system?
  -> if, not - use a numerical initial value solver:
     -choose a couple of strategically placed points in the plane (for example, equidistant on
      one of the axes) and just follow the field for positive and negative times
     -if start and end-points are known (such as with a dipole), maybe a boundary value solver
      can be used? 
      -for the dipole, we know that lines start at the source and end at the sink
      -we may also have to select an initial direction
     -maybe the function object passed to the plotter that generates the field lines is the solver?
      or maybe i should pre-solve the equation and pass some sort of InterpolatingFunction object
      (makes more sense, since the plotting routines want random access functions)
-for the complex mapping w = f(z) = z^2 interpreted as vector field, this leads to the first order
 ODE system x' = x^2 - y^2, y' = 2*x*y where the prime denotes differentiation with respect to the 
 parameter t.
 wolfram alpha can solve this with the command:
 DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == 2 x[t] y[t]}, {x, y}, t]
 which reults in 2 solutions:

  x = -(e^(2 C1) (t - 2 C2))/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  y = 1/2 (e^(C1) + sqrt(e^(2 C1) - (4 e^(4 C1) (t - 2 C2)^2)/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))

  x = -(e^(2 C1) (t + 2 C2))/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
  y = 1/2 (e^(C1) - sqrt(e^(2 C1) - (4 e^(4 C1) (t + 2 C2)^2)/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))

  ...try plotting them for various values of C1, C2 ...maybe to simplify, define 
  k1 = e^C1, k2 = e^C2, k12 = k1^2 = e^(2 C1), etc.

  do the same for x' = x^2 - y^2, y' = -2*x*y - the additional minus sign accounts for complex 
  conjugation of the Polya vector field associated to f(z) = z^2:
  DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == -2 x[t] y[t]}, {x, y}, t]
  ...but wolfram alpha doesn't understand this - wtf - the only difference is a minus sign!
  ok - wolfram cloud can do it - and the result is *much* more complicated than without the minus
  sign - it's still strange that alpha says that it doesn't understand the question - it should
  say something like: the problem is too complicated for alpha

  can we find a scalar valued function of x,y whose gradient is our desired vector field, i.e. a 
  potential function for our vector field?






  -> integrate with respect to t


  so it seems we
  must partially integrate f and g to obtain: 
  x(t) = integral_a^t f(x,y) dx, y(t) = integral_b^t g(x,y) dy
  for some constants a,b for the lower limits...is that correct? it seems reasonable to have two
  free parameters a,b since for each point on the plane, there should be exactly one field line
  that passes through it - so if we pick apoint x,y in the plane, we should be able to adjust a,b
  to find a field line through it (however that same field line passes through many points..hmm)




-how would we represent a scalar field in 3D? maybe as (semi-transparent) spheres with a size 
 representing the value? or maybe the transparency should represent the value? or both? ..or maybe
 cubes instead of spheres?

idea to visualize a rank-2 tensor field in 2D (i.e. a 2x2 matrix-field defined in the xy-plane)
-at each sample point, draw a small 2x2 square and color the 4 quadrants of that square according 
 to a colormap
-the size of squares may be such that when the sampling distances are dx, dy, a square centered at 
 x,y may extend from x-dx/2 to x+dx/2 in the x-direction and from y-dy/2 to y+dy/2 in the 
 y-direction - but we may artificially shrink these squares to have some margin between them
-a constant tensor field given by the 2x2 identity matrix will result in diagonal stripes of 
 high-value from top-left to bottom-right
-maybe it can be done via the matrix dataset and image graphing functions (similar to heat-maps 
 or spectrograms)
-maybe several such patterns with different sampling intervals can be overlaid to convey coarse
 and fine structure...maybe the coarse ones should be smoothed - or better: evaluated at an 
 oversampled grid and accumulated
 
-maybe try it with the Jacobian of f(z) = z^2 - the pattern should show symmetries due to the 
 Cauchy-Riemann conditions, try also the metric tensor of polar coordinates


-todo: draw a trefoil-knot, a mobius-strip and a trefoil-mobius-knot
 -for a pramatrization of the mobius-strip, see here: https://www.youtube.com/watch?v=dz7y7mFLW3U
 -for the mobius knot, replace the circle with the trefoil knot


*/