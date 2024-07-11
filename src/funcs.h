#ifndef GUARD_funcs_h
#define GUARD_funcs_h

#include <RcppArmadillo.h>
#include <cmath>
#include <iostream>
#include "tree.h"
#include "info.h"
#include "rand_gen.h"
#include "ProbHypers.h"

inline double logsumexp(const double &a, const double &b){
    return a<b? b + log(1.0 + exp(a-b)): a + log(1.0 + exp(b-a));
}

using std::cout;
using std::endl;

// define log2pi
#define LTPI 1.837877066

#define LPI 1.14473

// define pi
#define PI 3.14159265359

typedef std::vector<std::vector<int> > lookup_t;

lookup_t make_lookup(Rcpp::IntegerMatrix lookup_table, Rcpp::IntegerVector cx);

void impute_x(int v,
              std::vector<int>& mask,
              int n, xinfo& xi, std::vector<double>& x, std::vector<vector<int> >&x_cat,
              std::vector<int>& cx, std::vector<int>& offsets, std::vector<int>& x_type,
              std::vector<tree>& t, std::vector<double>& y, double& sigma, RNG& rng);

//normal density
double pn(double x, double m, double v);

// discrete distribution draw
int rdisc(double *p, RNG& gen);

// evaluate tree on grid xinfo, and write
void grm(tree&tr, xinfo& xi, std::ostream& os);

// check whether a node has variables it can split on
bool cansplit(tree::tree_p n, xinfo& xi);

// compute prob birth
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots);

// find variables can split on and store in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi, std::vector<size_t>& goodvars);

void getinternalvars(tree::tree_p n, xinfo& xi, std::vector<size_t>& goodvars);

int getnumcuts(tree::tree_p n, xinfo& xi, size_t var);

//-------------------
// get prob a node grows, 0 if no good vars, a/(1+d)^b else
double pgrow(tree::tree_p n, xinfo&xi, pinfo& pi);

// Calibart
void getLU(tree::tree_p pertnode, xinfo& xi, int* L, int* U);

void getpertLU(tree::tree_p pertnode, size_t pertvar, xinfo& xi, int* L, int* U);

//-------------------
// suff statistics for all bottom nodes
void allsuff(tree& x, xinfo& xi, dinfo& di, double* weight, tree::npv& bnv, std::vector<sinfo>& sv);
// counts for all bottom nodes
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di);
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv);

// update counts to reflect observation i?
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, int sign);

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, int sign);

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, std::map<tree::tree_cp, size_t>& bnmap, int sign);

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, std::map<tree::tree_cp, size_t>& bnmap, int sign, tree::tree_cp &tbn);
//-------------------

// check min leaf size
bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo& di);

//-------------------
// get sufficient stat for children (v,c) of node nx in tree x
void getsuffBirth(tree& x, tree::tree_cp nx, size_t v, size_t c, size_t p_mu, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr);

// get sufficient stat for pair of bottom children nl, nr, in tree x
void getsuffDeath(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr);

//---------------------
// log integrated likelihood
double loglike(double n, double sy, double sigma, double tau);
double loglikelg(double sdelta, double sy, double lg_alpha, double lg_beta);

double loglike_mvn(double n, arma::vec sy, arma::mat covsigma, arma::mat covprior);
//log pri of Tree
double logPriT(tree::tree_p x, xinfo& xi, pinfo& pi);
// fit
void fit(tree& t, xinfo& xi, dinfo& di, std::vector<double>& fv);

void fit(tree& t, xinfo& xi, dinfo& di, double* fv);

void predict(std::vector<tree>& tv, xinfo& xi, dinfo& di, double* fp);

// template function?
// one tree nodes mean?
template<class T>
std::vector<double> fit_i(T i, tree& t, xinfo& xi, dinfo& di)
{
    double *xx;
    /*
    std::vector<double> fv;//double fv = 0.0;
    tree::tree_cp bn;
    xx = di.x + i*di.p; //what for?
    bn = t.bot(xx,xi); // find leaf
    fv = bn->getmu();*/
    std::vector<double> fv(di.p_y);//double fv = 0.0;
    std::vector<double> tempsum(di.p_y);
    tree::tree_cp bn;
    xx = di.x + i*di.p;
    
    bn = t.bot(xx,xi);
    //fv.resize(bn->getp_mu()); tempsum.resize(fv.size());
    if (fv.size() != bn->getmu().size()) {
        Rcpp::Rcout << "unmatched size for mu while fitting the current tree\n";
        return fv;
    }
    //fv += bn->getmu();
    std::transform(fv.begin(), fv.end(), bn->getmu().begin(), tempsum.begin(), std::plus<double>());
    fv = tempsum;
    std::fill(tempsum.begin(), tempsum.end(), 0);
    
    return fv;
}

// forest nodes mean?
template<class T>
std::vector<double> fit_i(T i, std::vector<tree>& t, xinfo& xi, dinfo& di)
{
    double *xx;
    std::vector<double> fv(di.p_y);//double fv = 0.0;
    std::vector<double> tempsum(di.p_y);
    tree::tree_cp bn;
    xx = di.x + i*di.p;
    for (size_t j=0; j<t.size(); ++j) {
        bn = t[j].bot(xx,xi);
        //fv.resize(bn->getp_mu()); tempsum.resize(fv.size());
        if (fv.size() != bn->getmu().size()) {
            Rcpp::Rcout << "unmatched size for mu while fitting the " << j+1 << "th tree\n";
            return fv;
        }
        //fv += bn->getmu();
        std::transform(fv.begin(), fv.end(), bn->getmu().begin(), tempsum.begin(), std::plus<double>());
        fv = tempsum;
        std::fill(tempsum.begin(), tempsum.end(), 0);
    }
    return fv;
}

template<class T>
std::vector<double> fit_i_mult(T i, std::vector<tree>& t, xinfo& xi, dinfo& di)
{
    double *xx;
    std::vector<double> fv(di.p_y);//double fv = 0.0;
    std::vector<double> tempprod(di.p_y);
    tree::tree_cp bn;
    xx = di.x + i*di.p;
    for (size_t j=0; j<t.size(); ++j) {
        bn = t[j].bot(xx,xi);
        if (fv.size() != bn->getmu().size()) {
            Rcpp::Rcout << "unmatched size for mu while fitting the " << j+1 << "th tree\n";
            return fv;
        }
        //fv *= bn->getmu();
        std::transform(fv.begin(), fv.end(), bn->getmu().begin(), tempprod.begin(), std::multiplies<double>());
        fv = tempprod;
        std::fill(tempprod.begin(), tempprod.end(), 0);
    }
    return fv;
}

//---------------------
//partition

void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv);

// draw all bottom node mu's
#ifdef MPIBART

void MPImasterdrmu(tree& t, xinfo& xi, pinfo&pi, RNG& gen, size_t numslaves);
#else
void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double* weight, RNG& gen, arma::mat Covsig, arma::mat Covprior);
void drmu_withlg(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double* weight, RNG& gen, arma::mat Covsig, arma::mat Covprior, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu);
void drphi(tree& t, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
#endif

//---------------------
// write xinfo to screen
void prxi(xinfo& xi);

//---------------
//make xinfo
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc);
// get min/max for p predictors needed to make cutpoints
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx);
// make xinfo given min/max
void makexinfominmax(size_t p, xinfo&xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx);

// fucntions for updating mixture quatities
void updateLabels(int* labels, double* mixprop, double* locations, double* resid, double sigma, size_t nobs, size_t nmix, RNG& gen);
void updateMixprp(double* mixprop, double* mass, int* labels, int* mixcnts, size_t nmix, double psi1, double psi2, RNG& gen);
void updateLocations(double* locations, double* mixprop, double* resid, int* labels, int* mixcnts, double sigma, double prior_sigsq, size_t nobs, size_t nmix, RNG& gen);
void updateIndivLocations (double* indiv_locations, double* locations, int* labels, size_t nobs, size_t nmix);
bool CheckRule(tree::tree_p n, xinfo& xi, size_t var);
// function for coxph
//void riskset(std::vector<double> &yori, std::vector<double> &h, std::vector<double> &yord, std::vector<double> &delta, double* weight);
void riskset(std::vector<double> &yori, std::vector<double> &h0, std::vector<double> &hintv, std::vector<double> &whichinv, double* weight);
size_t sample_class(const std::vector<double> &probs, RNG& gen);
void UpdateS(dinfo& di, pinfo& pi, RNG& gen, std::vector<size_t> &ivcnt, std::vector<double> &S);
double log_sum_exp(const std::vector<double> &x);

#ifdef MPIBART
//MPI calls
void MPImasterallsuff(tree& x, tree::npv& bnv, std::vector<sinfo>& sv, size_t numslaves);
void MPIslaveallsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv);
void MPIslavedrmu(tree& t, xinfo& xi, dinfo& di);
void MPImastergetsuff(tree::tree_cp nl, tree::tree_cp nr, sinfo &sl, sinfo &sr, size_t numslaves);
void MPImastergetsuffvc(tree::tree_cp nx, size_t v, size_t c, xinfo& xi, sinfo& sl, sinfo& sr, size_t numslaves);
void MPImastersendbirth(tree::tree_p nx, size_t v, size_t c, double mul, double mur, size_t numslaves);
void MPImastersenddeath(tree::tree_p nx, double mu, size_t numslaves);
void MPImastersendnobirthdeath(size_t numslaves);
void MPIslaveupdatebirthdeath(tree& x);
void MPIslavegetsuff(tree& x, xinfo& xi, dinfo& di);
void makepred(dinfo dip, xinfo &xi, std::vector<tree> &t, double *ppredmean);
void makeypred(dinfo dip, xinfo &xi, std::vector<tree> &t, double sigma, double *ppredmean);
void makepostpred(dinfo dip, xinfo &xi, std::vector< std::vector<tree> > &t, double *postp);
void makepostpred2(dinfo dip, xinfo &xi, std::vector< std::vector<tree> > &t, double *postp, double *postp2);
double logcalp(std::vector<double> &theta, dinfo dip, xinfo &xi, std::vector<tree> &t, double sigmae, double sigma, size_t pth, size_t myrank);
#endif


// ProbHypers from SBART-MFM (Linero and Yang, 2018): https://github.com/theodds/SoftBART/blob/MFM/src/soft_bart.h
/*
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
    
    std::map<std::pair<size_t, size_t>, double> log_V;
    std::vector<double> counts;
    std::vector<double> log_prior;
};
*/
void AddTreeCounts(ProbHypers& hypers, tree::tree_p x);
void SubtractTreeCounts(ProbHypers& hypers, tree::tree_p x);
double cutpoint_likelihood(tree::tree_p x, xinfo& xi);
void GenBelow(tree::tree_p x, pinfo& pi, dinfo& di, xinfo& xi, double* phi, ProbHypers& hypers, RNG& gen, bool vs);
double BSinvTrigamma(double lower, double upper, double x);
double bspline(double x, int i, int degree, const std::vector<double>& knots);
double GetHZD(double x, int degree, const std::vector<double>& knots, double* gamma);
double GetCumhzd(double x, int degree, size_t ngrids, const std::vector<double>& knots, double* gamma);
double gammaFC(double rho, double eps, int degree, size_t ngrids, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma);
void gammaMH(double rho, double eps, int degree, size_t ngrids, double stepsize, double* MHcounts_gamma, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma, RNG& gen);
void gammaMHcons(double rho, double eps, int degree, size_t ngrids, double stepsize, double* MHcounts_gamma, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma, RNG& gen);
#endif /* funcs_h */
