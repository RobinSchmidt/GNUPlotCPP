#include "GNUPlotter.h"
//#include <math.h>
using namespace std;

#define M_PI 3.14159265358979323846

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

void complexExperiment()
{
  // Set up range and umber of sampling points for real and imaginary part:
  double rMin = -3; double rMax = +3; int Nr = 31;
  double iMin = -3; double iMax = +3; int Ni = 31;

  // Define the complex function w = f(z) = z^2 as example complex function:
  function<complex<double>(complex<double>)> f;
  f = [] (complex<double> z) { return z*z; };
  //f = [] (complex<double> z) { return z*z*z; };
  //f = [] (complex<double> z) { return z*z*z*z; };
  //f = [] (complex<double> z) { return 1./z; };                 // 1st order pole at z=0
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



// maybe wrap this code into a class "VectorFieldZedSquared" ..or just ZedSquared...maybe have 
// similar functions ZedCubed, ZedInverse or OneOverZed, ExpZed, SineZed, etc.

// x-component of 1st solution to the first order ODE system x' = x^2 - y^2, y' = 2*x*y which can 
// be obtained by wolfram alpha via:
// DSolve[ {x'[t] == x[t]^2 - y[t]^2, y'[t] == 2 x[t] y[t]}, {x, y}, t] 
// which gives two solutions
double squareFieldX1(double t, double C1, double C2)
{
  double k1 = exp(C1);   // e^C1
  double k2 = k1*k1;     // e^(2 C1) = k1^2
  double x  = -(k2 * (t-2*C2)) / (1+4*k2*C2*C2 + k2*t*t - 4*k2*t*C2);
  return x;
  // x = -(e^(2 C1) (t - 2 C2))/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2);
}
double squareFieldY1(double t, double C1, double C2)
{
  double k1 = exp(C1);   // e^C1
  double k2 = k1*k1;     // e^(2 C1) = k1^2
  double a  = 1+4*k2*C2*C2 - 4*k2*t*C2 + k2*t*t;
  double b  = t - 2*C2;
  double s  = k2-(4*k2*k2*b*b) / a*a;
  double y  = (1./2) * (k1 + sqrt(max(0.0,s)));
  return y;
  // y = 1/2 (e^(C1) + sqrt(e^(2 C1) - (4 e^(4 C1) (t - 2 C2)^2)/(1 + e^(2 C1) t^2 - 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
}
void squareFieldParamLimits1(double c1, double c2, double* tMin, double* tMax)
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
double squareFieldX2(double t, double C1, double C2)
{
  double k1 = exp(C1);   // e^C1
  double k2 = k1*k1;     // e^(2 C1) = k1^2
  double x  = -(k2 * (t+2*C2)) / (1+4*k2*C2*C2 + k2*t*t + 4*k2*t*C2);
  return x;
  // x = -(e^(2 C1) (t + 2 C2))/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)
}
double squareFieldY2(double t, double C1, double C2)
{
  double k1 = exp(C1);   // e^C1
  double k2 = k1*k1;     // e^(2 C1) = k1^2
  double a  = 1+4*k2*C2*C2 + 4*k2*t*C2 + k2*t*t;
  double b  = t + 2*C2;
  double s  = k2-(4*k2*k2*b*b) / a*a;
  double y  = (1./2) * (k1 - sqrt(max(0.0,s)));
  return y;
  // y = 1/2 (e^(C1) - sqrt(e^(2 C1) - (4 e^(4 C1) (t + 2 C2)^2)/(1 + e^(2 C1) t^2 + 4 e^(2 C1) t C2 + 4 e^(2 C1) (C2)^2)^2))
}
// todo: refactor to avoid code duplication...maybe try to avoid recomputations - but to fully 
// avoid them, we may need to be able to pass a vector-valued function to the plotter instead of 
// two scalar valued functions (the computations for x and y have terms in common)

// maybe try to use natural parametrization (nach bogenlänge parametrisieren) - find a function to
// apply to t to find s...or actually we need t as function of s

void squareFieldParamLimits2(double c1, double c2, double* tMin, double* tMax)
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

void curveInVectorFieldExperiment()  // rename to zedSquaredVectorField
{
  // We plot the 2D vector field corresponding to the complex function f(z) = z^2 and also draw a
  // curve that represents a field line (todo: draw may field lines for various values of c1 (and
  // maybe c2))

  // user parameters:
  double c1 = 1.0, c2 = -0.5;             // field line parameters (select, which line is drawn)

  // create and set up plotter:
  GNUPlotter plt;                         // plotter object
  plt.clearCommandFile();                 // we don't want to use the default line styles/colors
  plt.setGrid();
  //plt.setGraphColors("FF0000", "0000FF"); // doesn't work - but default color cyan looks ok

  // local variables:
  std::function<double(double, double)> fx, fy; // vector field fx(x,y), fy(x,y)
  std::function<double(double)> gx, gy;         // field line functions x(t), y(t)
  double tMin, tMax;                            // limits for time parameter for field line
  // c1 controls size of the field lines but c2 seems to have no visible effect (maybe it controls
  // the speed and therefore the range tMin..tMax?)

  // add data for the 2 bivariate functions and commands for plotting the vector field:
  fx = [] (double x, double y) { return x*x - y*y; }; // x^2 - y^2 = Re{ z^2 }
  fy = [] (double x, double y) { return 2*x*y;     }; // 2xy       = Im{ z^2 }
  plt.addDataVectorField2D(fx, fy, 31, -3., +3., 31, -3., +3.);
  plt.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette notitle");
  plt.addCommand("set palette rgbformulae 30,31,32 negative");

  // add data and commands for the first half of the field line:
  squareFieldParamLimits1(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return squareFieldX1(t, c1, c2); };
  gy = [&] (double t) { return squareFieldY1(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, 401, tMin, tMax, true);
  plt.addGraph("index 1 using 2:3 with lines notitle"); // 2:3 bcs 1 is the parameter t

  // add data and commands for the second half of the field line:
  squareFieldParamLimits2(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return squareFieldX2(t, c1, c2); };
  gy = [&] (double t) { return squareFieldY2(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, 401, tMin, tMax, true);
  plt.addGraph("index 2 using 2:3 with lines notitle");

  plt.plot();
  // todo:
  // -plot muliple field lines
  // -add arrows and maybe something that let's use see the speed (maybe plot segments where the 
  //  particle is fast fainter - resembles an analog oscilloscope look)
}
// -how about equipotential lines? for this, we perhaps first should figure out how to draw several
//  curves on top of a scalar field in general

// todo: plot vector field of a polynomial with zeros placed at -1, +1, -i, +i, 0, maybe
// also plot vector fields of rational functions

/*
Ideas:
-move high-level vector-field to demos - make a demo showing a positive charge at +1 and a negative
 charge at -1 using the physically correct law for the electric field - make another plot with two
 negative and two positive charges - the two negative charges should look like 2 gravitational 
 fields - call it demoDipole



-plot 3D curves (trefoil knot)
-plot 3D vector fields
-plot curves on top of a 2D vector field - can be used to show equipotential curves and/or 
 integration paths

 weitz on visualization:
 https://www.youtube.com/watch?v=BhtnlKOC-0s&t=189s


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






*/