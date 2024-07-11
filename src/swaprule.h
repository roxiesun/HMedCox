#ifndef GUARD_swap_h
#define GUARD_swap_h

#include "rand_gen.h"
#include "info.h"
#include "tree.h"
#include "logging.h"

bool swaprule(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior);
bool swaprule_lg(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior);
bool swaprule_withvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs);
bool swaprule_lgwithvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs);

#endif /* swap_h */
