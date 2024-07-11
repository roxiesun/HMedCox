#ifndef GUARD_slice_h
#define GUARD_slice_h

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rand_gen.h"

using namespace Rcpp;
using namespace std;


NumericVector slice_sample_cpp(double (*logfn)(NumericVector, NumericVector, std::vector<double>&, std::vector<double>&, std::vector<double>&),
                        NumericVector params,
                        NumericVector x0,
                        std::vector<double>& storevec1,
                        std::vector<double>& storevec2,
                        std::vector<double>& storemu,
                        int steps,
                        double w,
                        double lower,
                        double upper);

double post_hcauchy(NumericVector x, NumericVector params, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu);

NumericVector hcsig_update (NumericVector x, NumericVector params, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu, RNG& gen);

/*
double SliceSample(double (*logfn)(double, double), double x0, double pars, double width, double ub, double lb, int miter)
{
    double y, xl, xr, x;
    y = logfn(x0, pars) + log(R::runif(0.0, 1.0));
    xl = x0 - width * R::runif(0.0, 1.0);
    xr = xl + width;
    for (int j = 0; j < miter; ++j) {
        if (xl < lb) { xl = lb; break; }
        else if (logfn(xl, pars) > y) xl -= width;
        else break;
    }
    for (int j = 0; j < miter; ++j) {
        if (xr > ub) { xr = ub; break; }
        else if (logfn(xr, pars) > y) xr += width;
        else break;
    }
    x = R::runif(0.0, 1.0) * (xr - xl) + xl;
    for (int j = 0; j < miter; ++j) {
        if (logfn(x, pars) > y) break;
        else {
            if (x > x0) xr = x;
            else xl = x;
            x = R::runif(0.0, 1.0) * (xr - xl) + xl;
            if (x > ub) x = ub;
            else if (x < lb) x = lb;
        }
    }
    return x;
}
*/

#endif /* slice_h */
