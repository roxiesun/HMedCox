// [[Rcpp::depends(RcppParallel)]]
#include <cmath>
#include "rand_gen.h"
#include "funcs.h"
#include "slice.h"
#include <map>
#include <RcppParallel.h>
#include <Rcpp.h>
#include <vector>
#ifdef MPIBART
#include "mpi.h"
#endif

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
                        double upper) {
    

    double u, r0, r1, logy, logz, logys;
    NumericVector x, xs, L, R;

    x = clone(x0);
    L = clone(x0);
    R = clone(x0);
    logy = logfn(x, params, storevec1, storevec2, storemu);

  

    for (int j = 0; j < x0.size(); j++) {
        
      // draw uniformly from [0, y]
        logz = logy + log(R::runif(0.0,1.0));//- rexp(1)[0];
        

      // expand search range
        u = R::runif(0.0,1.0) * w;
        L[j] = x[j] - u;
        R[j] = x[j] + w;
    
        for (size_t k=0; k < steps; ++k) {
            if (L[j] < lower) {
                L[j] = lower; break;
            }
            else if (logfn(L, params, storevec1, storevec2, storemu) > logz)
                L[j] -= w;
            else break;
        }
        
        for (size_t k=0; k < steps; ++k) {
            if (R[j] > upper) {
                R[j] = upper; break;
            }
            else if (logfn(R, params, storevec1, storevec2, storemu) > logz)
                R[j] += w;
            else break;
        }
        

        
        r0 = std::max(L[j], lower);
        r1 = std::min(R[j], upper);

        xs = clone(x);
        xs[j] = R::runif(0.0,1.0) * (r1 - r0) + r0;
        //logys = logfn(xs, params, storevec1, storevec2, storemu);
        for (size_t k = 0; k < steps; ++k) {
            //xs[j] = R::runif(0.0,1.0) * (r1 - r0) + r0;
            logys = logfn(xs, params, storevec1, storevec2, storemu);
            if (logys > logz) break;
            else {
                if (xs[j] > x0[j]) {
                    r1 = xs[j];
                }
                else r0 = xs[j];
                xs[j] = R::runif(0.0,1.0)* (r1 - r0) + r0;
                if (xs[j] > upper) {
                    xs[j] = upper;
                }
                else if (xs[j] < lower) xs[j] = lower;
            }
            
        }

        x = clone(xs);
        logy = logys;
    }

    return x;
}

double post_hcauchy(NumericVector x, NumericVector params, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu) {
    double mu = params[0];
    double sigma = params[1];
    double tmp = 0.0;
    double out = 0.0;
    //for (size_t i = 0; i < storevec1.size(); ++i) {
        //tmp +=  BSinvTrigamma(0.01, 1000, x[0]*x[0])*storemu[i] -  exp(R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])))*exp(storemu[i]) + BSinvTrigamma(0.01, 1000, x[0]*x[0])*R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])) - std::lgamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])) -  (storevec1[i] + BSinvTrigamma(0.01, 1000, x[0]*x[0]))*log(storevec2[i] + exp(R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])))) + std::lgamma(storevec1[i] + BSinvTrigamma(0.01, 1000, x[0]*x[0]));//(storevec1[i] + BSinvTrigamma(0.01, 1000, x[0]*x[0]))*storemu[i] - (storevec2[i] + exp(R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0]))))*exp(storemu[i]) + BSinvTrigamma(0.01, 1000, x[0]*x[0])*R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])) - std::lgamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])) -  (storevec1[i] + BSinvTrigamma(0.01, 1000, x[0]*x[0]))*log(storevec2[i] + exp(R::digamma(BSinvTrigamma(0.01, 1000, x[0]*x[0])))) + std::lgamma(storevec1[i] + BSinvTrigamma(0.01, 1000, x[0]*x[0]));//(storevec1[i] + 1.0/(x[0]*x[0]) + 0.5)*storemu[i] - (storevec2[i] + 1.0/(x[0]*x[0]))*exp(storemu[i]) + (1.0/(x[0]*x[0]) + 0.5)*log(1.0/(x[0]*x[0])) - std::lgamma(1.0/(x[0]*x[0]) + 0.5) -  (storevec1[i] + 1.0/(x[0]*x[0]) + 0.5)*log(storevec2[i] + 1.0/(x[0]*x[0])) + std::lgamma(storevec1[i] + 1.0/(x[0]*x[0]) + 0.5);
    //}
    if (x[0] > 0.) {
        out = tmp + log(2.) -LPI + log(sigma) - log((x[0] - mu)*(x[0] - mu) + sigma*sigma);
    } else {out = -INFINITY;}
    return out;
}


NumericVector hcsig_update (NumericVector x, NumericVector params, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu, RNG& gen) {
    
    //double current = x[0];
    //double proposed = x[0];
    NumericVector current = clone(x);
    NumericVector proposed = clone(x);
    double sum = 0.;
    for (size_t k=0; k<storemu.size(); ++k) {
        sum += exp(storemu[k]) - storemu[k];
    }

    
    double tmp = gen.exp(1/sum);
    proposed[0] = sqrt(1/tmp);
    
    double alpha = exp(post_hcauchy(proposed, params, storevec1, storevec2, storemu) - post_hcauchy(current, params, storevec1, storevec2, storemu)) * pow(proposed[0],3) / pow(x[0],3);
    
    double u = gen.uniform();
    if (u < alpha) {
        x[0] = proposed[0];
    }
    return x;
    
}
