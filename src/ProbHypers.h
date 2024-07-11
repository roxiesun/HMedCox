#ifndef GUARD_ProbHypers_h
#define GUARD_ProbHypers_h

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rand_gen.h"


struct ProbHypers {
    bool use_counts;
    double dirichlet_mass;
    double log_mass;
    //logV_B(t), n for B and t for t, in the Gibbs prior
    double calc_log_v(size_t n, size_t t) {
        std::pair<size_t, size_t> nt(n,t);
        
        std::map<std::pair<size_t, size_t>, double>::iterator iter = log_V.find(nt);
        if (iter != log_V.end()) {
            return iter->second;
        }
        
        size_t D = log_prior.size();
        std::vector<double> log_terms;
        for (size_t k = t; k <= D; k++) {
            double log_term = log_prior[k-1];
            log_term += R::lgammafn(k+1) - R::lgammafn(k-t+1);
            log_term += R::lgammafn(dirichlet_mass * k) - R::lgammafn(dirichlet_mass * k + n);
            log_terms.push_back(log_term);
        }
        log_V[nt] = log_sum_exp(log_terms);
        return log_V[nt];
    }
    
    ProbHypers(double d_mass, std::vector<double> &log_p, bool use_c): use_counts(use_c), dirichlet_mass(d_mass), log_prior(log_p) {
        log_V.clear();
        log_mass = log(d_mass);
        counts = std::vector<double>(log_p.size(), 0.);
    }
    ProbHypers() {}
    
    size_t SampleVar(RNG& gen) {
        size_t var_idx = counts.size() - 1;
        double U = gen.uniform();
        
        if (!use_counts) {
            var_idx = floor(U*counts.size());
            return var_idx;
        }
        
        double cumsum = 0.0;
        size_t K = counts.size();
        size_t num_branches = std::accumulate(counts.begin(), counts.end(), 0.);
        size_t num_active = 0;
        for (size_t k = 0; k < counts.size(); ++k) {
            num_active += (counts[k] > 0) ? 1 : 0;
        }
        
        for (size_t k = 0; k < K; ++k) {
            if (counts[k] == 0) {
                double tmp = calc_log_v(num_branches + 1, num_active + 1);
                tmp -= calc_log_v(num_branches, num_active);
                tmp += log(dirichlet_mass + counts[k]);
                cumsum += exp(tmp);
            } else {
                double tmp = calc_log_v(num_branches + 1, num_active);
                tmp -= calc_log_v(num_branches, num_active);
                tmp += log(dirichlet_mass + counts[k]);
                cumsum += exp(tmp);
            }
            if (U < cumsum) {
                var_idx = k;
                break;
            }
        }
        return var_idx;
    }
    
    size_t ResampleVar(size_t var, RNG& gen) {
        counts[var] -= 1;
        size_t sampled_var = SampleVar(gen);
        counts[sampled_var] += 1;
        return sampled_var;
    }
    
    void SwitchVar(size_t v_old, size_t v_new) {
        counts[v_old] -= 1;
        counts[v_new] += 1;
    }
    
    double log_sum_exp(const std::vector<double> &x)
    {
        double M = *std::max_element(std::begin(x), std::end(x));
        double tmp = 0;
        for (size_t k=0; k<x.size(); ++k) {
            tmp += exp(x[k] - M);
        }
        return M + log(tmp);
    }
    
    std::map<std::pair<size_t, size_t>, double> log_V;
    std::vector<double> counts;
    std::vector<double> log_prior;
};


#endif /* ProbHypers_h */
