#ifndef GUARD_draw_prior_h
#define GUARD_draw_prior_h

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rand_gen.h"
#include "ProbHypers.h"

bool draw_prior_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers);

bool draw_prior_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers);


#endif /* draw_prior_h */
