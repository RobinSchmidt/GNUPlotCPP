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

void addFieldLine(GNUPlotter& plt, double c1, bool flipY = false) 
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
  squareFieldParamLimits1(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        squareFieldX1(t, c1, c2); };
  gy = [&] (double t) { return sign * squareFieldY1(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3); 

  // add data and commands for the second half of the field line:
  s2 = to_string(2*numFieldLines+2);
  squareFieldParamLimits2(c1, c2, &tMin, &tMax);
  gx = [&] (double t) { return        squareFieldX2(t, c1, c2); };
  gy = [&] (double t) { return sign * squareFieldY2(t, c1, c2); };
  plt.addDataCurve2D(gx, gy, numPoints, tMin, tMax);
  plt.addGraph(s1+s2+s3);

  numFieldLines++;
}

void curveInVectorFieldExperiment()  // rename to zedSquaredVectorField
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
    addFieldLine(plt, c, false);
    addFieldLine(plt, c, true);
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


void pendulumPhasePortrait() // move to Demos
{
  // physical parameters:
  double mu = 0.15; // damping constant
  double g  = 1;    // gravitational pull/acceleration
  double L  = 1;    // length of pendulum


  std::function<double(double, double)> fx, fy; // vector field fx(x,y), fy(x,y)
  fx = []  (double x, double y) { return y; };
  fy = [&] (double x, double y) { return -mu*y - (g/L)*sin(x); };
  // https://www.youtube.com/watch?v=p_di4Zn4wz4 3blue1brown video about this sort of plot



  GNUPlotter plt;

  // vector field arrows:
  plt.addVectorField2D(fx, fy, 51, -10., +10., 41, -4., +4.);

  //plt.addDataVectorField2D(fx, fy, 51, -10., +10., 41, -4., +4.);
  //plt.addGraph("index 0 using 1:2:3:4:5 with vectors head filled size 0.08,15 ls 2 lc palette notitle");
    // maybe have a function addGraphVectorField2D and/or let the addData function have a bool
    // parameter that lets the graph be added automatically

  // 3 trajectories:
  plt.setGraphColors("209050");   // trajectory color
  plt.addDataFieldLine2D(fx, fy, -9.9, 4.0, 0.1, 1000, 10);
  plt.addGraph("index 1 using 1:2 with lines lt 1 notitle");
  plt.addDataFieldLine2D(fx, fy, -4.0, 1.5, 0.1, 1000, 10);
  plt.addGraph("index 2 using 1:2 with lines lt 1 notitle");
  plt.addDataFieldLine2D(fx, fy, 5.0, -3.0, 0.1, 1000, 10);
  plt.addGraph("index 3 using 1:2 with lines lt 1 notitle");
    // try to get rid of the addGraph commands here, too - maybe have a function
    // addGraphFieldLine2D ...check out, how GNUPlotter deals with the dataInfo member -  i think, 
    // it's used only, if graphDescriptors is empty? if so, it should be cleanly possible to 
    // combine addData and addGraph into a single function


  plt.addCommand("set palette rgbformulae 30,31,32 negative");
  plt.addCommand("set xrange [-10.5:10.5]");
  plt.addCommand("set yrange [-4.5:4.5]");
  plt.addCommand("set xlabel \"Angle {/Symbol q}\"");
  plt.addCommand("set ylabel \"Angular velocity {/Symbol w}\"");
  plt.addCommand("set xtics pi");
  plt.addCommand("set format x '%.0P{/Symbol p}'");
  plt.setPixelSize(1000, 500); 
  plt.plot();

  // greek and/or latex letters:
  // https://sourceforge.net/p/gnuplot/discussion/5925/thread/bc8a65fe/
  // http://www.gnuplot.info/files/tutorial.pdf
  // https://tex.stackexchange.com/questions/119518/how-can-add-some-latex-eq-or-symbol-in-gnuplot
  // https://stackoverflow.com/questions/28964500/math-in-axes-labels-for-gnuplot-epslatex-terminal-not-formatted-correctly

  // tics at multiples of pi:
  // http://www.gnuplotting.org/tag/tics/  // ...it also says something about multiplots
  // http://www.gnuplotting.org/set-your-tic-labels-to-use-pi-or-to-be-blank/


  // -maybe try a black background (and invert the colormap)


}


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






*/