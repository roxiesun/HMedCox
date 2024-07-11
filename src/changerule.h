#ifndef GUARD_changerule_h
#define GUARD_changerule_h

#include "rand_gen.h"
#include "info.h"
#include "tree.h"
#include "logging.h"
#include "ProbHypers.h"

bool changerule(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
bool changerule_lg(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
bool changerule_withvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);
bool changerule_lgwithvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);

#endif /* changerule_h */

