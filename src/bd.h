#ifndef GUARD_bd_h
#define GUARD_bd_h

#include "rand_gen.h"
#include "info.h"
#include "tree.h"
#include "logging.h"
#include "ProbHypers.h"

#ifdef MPIBART
bool bd(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
#else
bool bd(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
bool bd_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);
bool bd_lg(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
bool bd_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);
#endif

#endif /* bd_h */
