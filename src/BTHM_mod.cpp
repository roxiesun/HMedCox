#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "rand_gen.h"
#include "tree.h"
#include "info.h"
#include "funcs.h"
#include "bd.h"
#include "logging.h"
#include "changerule.h"
#include "swaprule.h"
#include "draw_prior.h"
#include "slice.h"

using namespace Rcpp;


/* model: Bayesian tree ensembles for heterogeneous mediation
 Mediator part:
 M_ik = ups_k(x_i) + tau_k(x_i)A_i + \eps_ik
 
 Cox PH part:
 H(t|) = h0(t)exp(ups(X_i,M_i) + \tau(x_i)A_i)
 
 dimension parameters: p_m,p_x,number of pieces for h0?*/


//[[Rcpp::interfaces(r, cpp)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::export]]

List BTHMRcpp(NumericVector y_, NumericVector a_, NumericVector w_, NumericVector x_con_, NumericVector x_mod_, NumericVector x_coxcon_, NumericVector m_,  NumericVector status_, List x_con_info_list, List x_mod_info_list, List x_coxcon_info_list, NumericVector intv_, int burn, int nd, int thin, int ntree_coxcon, int ntree_mod, int ntree_con, double nu, double con_alpha, double con_beta, double mod_alpha, double mod_beta, double coxcon_alpha, double coxcon_beta, NumericVector kappa_, NumericVector sigest_, NumericVector zeta_, NumericVector truehzd_, NumericVector hcsig_coxcon_, NumericVector hcsig_mod_, CharacterVector treef_con_name_, CharacterVector treef_mod_name_, CharacterVector treef_coxcon_name_, double t_survp, int printevery=100,  double trt_init=0.5, int nknots = 20, bool verbose_sigma=false, bool vs = false)
{

    std::string treef_con_name = as<std::string>(treef_con_name_);
    std::ofstream treef_con(treef_con_name.c_str());

    std::string treef_mod_name = as<std::string>(treef_mod_name_);
    std::ofstream treef_mod(treef_mod_name.c_str());
    
    std::string treef_coxcon_name = as<std::string>(treef_coxcon_name_);
    std::ofstream treef_coxcon(treef_coxcon_name.c_str());

    
    RNGScope scope;
    RNG gen; // one random number generator used in all draws
    
    
    Logger logger = Logger();
    char logBuff[100];

    bool log_level = false;

    logger.setLevel(log_level);
    logger.log("=================================");
    logger.log("Starting up BTHM: ");
    logger.log("=================================");
    if (log_level) {
        logger.getVectorHead(y_, logBuff);
        Rcout << "y: " << logBuff << "\n";
        logger.getVectorHead(a_, logBuff);
        Rcout << "a: " << logBuff << "\n";
        logger.getVectorHead(w_, logBuff);
        Rcout << "w: " << logBuff << "\n";
    }
    
    //logger.log("Algorithm is Weighted");
    logger.log("");

    /* **** Read format y **** */
    std::vector<double> yobs, y, a;
    std::vector<double> delta, deltaA;// y is for true survival time, yobs is the observed event/censoring time, delta is the event indicator
    double miny = INFINITY, maxy = -INFINITY;
    sinfo allys;// suff stat for all y, used to initialized the barts
    //allys.sy.resize(1); allys.n.resize(1);
    int num_censored = 0, num_treated = 0;
    
    for (NumericVector::iterator it=y_.begin(); it!=y_.end(); ++it) {
        yobs.push_back(*it);
        y.push_back(*it);
        if (*it<miny) miny=*it;
        if (*it>maxy) maxy=*it;
        allys.sy[0] += *it; //\sum y
    }
    
    // read in status: 1 for event, 0 for censored
    for (NumericVector::iterator it = status_.begin(); it!=status_.end(); ++it) {
        delta.push_back(*it);
        num_censored += (1 - *it);
    }
    
    for (NumericVector::iterator it = a_.begin(); it!=a_.end(); ++it) {
        a.push_back(*it);
        num_treated += *it;
    }
    
    size_t n = y.size();
    allys.n[0] = (double)n;
    double ybar = allys.sy[0]/(double)n; // sample mean
    
    for (size_t i=0; i<n; ++i) {
        deltaA.push_back(a[i]*delta[i]);
    }
    
    /* **** Read format m **** */
    std::vector<double> m;
    //double minm = INFINITY, maxm = -INFINITY;
    sinfo allms; // suff stat for all m, used to initialized the barts
    size_t p_m = m_.size()/n;
    Rcout << "Considering " << p_m << " mediators." << std::endl;
    allms.ysize = p_m; allms.sy.resize(p_m); allms.n.resize(p_m);
    
    for (NumericVector::iterator it=m_.begin(); it!=m_.end(); ++it) {
        m.push_back(*it);
    }
    
    for (size_t k=0; k<n; ++k) {
        for (size_t j=0; j<p_m; ++j) {
            allms.sy[j] += m[k*p_m+j];
        }
    }
    std::vector<double> mbar(p_m);
    for (size_t k=0; k<p_m; ++k) {mbar[k] = allms.sy[k]/(double)n;}
    
    std::vector<double> mny;
    mny.resize(y.size()+m.size());
    for (size_t k=0; k<n; ++k) {
        for (size_t j=0; j<p_m; ++j) {
            mny[k*(p_m+1)+j] = m[k*p_m+j];
        }
        mny[k*(p_m+1)+p_m] = y[k];
    }
    
    /* **** Read format weights **** */
    double* w = new double[n];

    for (int j=0; j<n; j++) {
        w[j] = w_[j];
    }
    
    /* **** Read format x_con **** */
    std::vector<double> x_con;
    for (NumericVector::iterator it=x_con_.begin(); it!=x_con_.end(); ++it) {
        x_con.push_back(*it);
    }
    size_t p_con = x_con.size()/n;
    std::vector<size_t> nv_con(p_con,0.);

    Rcout << "Using " << p_con << " control variables." << std::endl;

    // cutpoints;
    xinfo xi_con;
    xi_con.resize(p_con);
    for (int i=0; i<p_con; ++i) {
        NumericVector tmp = x_con_info_list[i];
        std::vector<double> tmp2;
        for (size_t j=0; j<tmp.size(); ++j) {
            tmp2.push_back(tmp[j]);
        }
        xi_con[i] = tmp2;
        tmp2.clear();
    }
    
    /* **** Read format x_mod **** */
    int ntrt = 0;
    for (size_t i=0; i<n; ++i) {
        if (a_[i]>0) ntrt +=1; // Note: binary treatment!!
    }
    std::vector<double> x_mod;
    for (NumericVector::iterator it=x_mod_.begin(); it!=x_mod_.end(); ++it) {
        x_mod.push_back(*it);
    }
    size_t p_mod = x_mod.size()/n;
    std::vector<size_t> nv_mod(p_mod,0.);

    Rcout << "Using " << p_mod << " potential effect moderators." << std::endl;

    // cutpoints
    xinfo xi_mod;

    xi_mod.resize(p_mod);
    for (int i=0; i<p_mod; ++i) {
        NumericVector tmp = x_mod_info_list[i];
        std::vector<double> tmp2;
        for (size_t j=0; j<tmp.size(); ++j) {
            tmp2.push_back(tmp[j]);
        }
        xi_mod[i] = tmp2;
        tmp2.clear();
    }
    
    /* **** Read format x_coxcon **** */
    // which is actually (x_i,M_i1,...,M_ik)
    std::vector<double> x_coxcon;
    for (NumericVector::iterator it=x_coxcon_.begin(); it!=x_coxcon_.end(); ++it) {
        x_coxcon.push_back(*it);
    }
    size_t p_coxcon = x_coxcon.size()/n;// + p_m;
    std::vector<size_t> nv_coxcon(p_coxcon,0.);
    
    Rcout << "Using " << p_coxcon << " predictors for the cox model." << std::endl;

    // cutpoints;
    xinfo xi_coxcon; //note, M info should also be contained in the x_coxcon_info_list!
    xi_coxcon.resize(p_coxcon);
    for (int i=0; i<p_coxcon; ++i) {
        NumericVector tmp = x_coxcon_info_list[i];
        std::vector<double> tmp2;
        for (size_t j=0; j<tmp.size(); ++j) {
            tmp2.push_back(tmp[j]);
        }
        xi_coxcon[i] = tmp2;
        tmp2.clear();
    }
    
    /* **** Set up the model **** */

    // trees
    std::vector<double> mu_init (p_m+1, trt_init/(double)ntree_mod);
    std::vector<tree> t_mod(ntree_mod);
    for (size_t i=0; i<ntree_mod; i++) {
        t_mod[i].setp_mu(p_m+1);
        t_mod[i].setmu(mu_init);
    }
    
    mu_init.resize(p_m);
    for (size_t k=0; k<p_m; ++k) {mu_init[k] = mbar[k]/(double)ntree_con;}
    std::vector<tree> t_con(ntree_con);
    for (size_t i=0; i<ntree_con; i++) {
        t_con[i].setp_mu(p_m);
        t_con[i].setmu(mu_init);
    }
    
    mu_init.resize(1); mu_init[0] = 0./(double)ntree_coxcon; //ybar/(double)ntree_coxcon;
    std::vector<tree> t_coxcon(ntree_coxcon);
    for (size_t i=0; i<ntree_coxcon; i++) {
        t_coxcon[i].setp_mu(1);
        t_coxcon[i].setmu(mu_init);
    }

    // prior
    // scale parameters for b, the modifier part
    //double bscale_prec = 2;
    double bscale0 = -1.0; // are the two bscales ps ?
    double bscale1 = 1.0;

    //double mscale_prec = 1.0; // half cauchy prior for muscale
    double mscale = 1.0;
    //double delta_con = 1.0; // half normal prior for tau scale, the homo treatment effect
    //double delta_mod = 1.0;

    pinfo pi_mod;
    pi_mod.pbd = 0.7;//0.5;
    pi_mod.pb = 0.35;//0.25;
    pi_mod.pchange = 0.3;//0.4;
    pi_mod.pswap = 0.0;//0.1;
    
    pi_mod.alpha = mod_alpha;
    pi_mod.beta = mod_beta;
    //pi_mod.tau = zeta/(8*sqrt(ntree_mod));  // sigma_mu, variance on leaf parameters, =zeta/(2k*sqrt(ntree))
    pi_mod.tau = new double [p_m];
    pi_mod.sigma = new double [p_m];
    for (size_t k=0; k<(p_m); ++k) {
        pi_mod.tau[k] = zeta_[k]/(8*sqrt((double)ntree_mod));
        pi_mod.sigma[k] = sigest_[k];
    }
    //pi_mod.sigma = sigest; // residual variance is \sigma^2_y / bscale^2
    double hcsigma_1 = 0.25;//hcsig_mod_[0]; // gen.halfcauchy(1.5/sqrt((double)ntree_mod));//hcsig_mod_[0]; //
    pi_mod.lg_alpha = 1.0/(hcsigma_1*hcsigma_1) + 0.5;
    pi_mod.lg_beta = 1.0/(hcsigma_1*hcsigma_1);
    // or use the trigamma fucntion?
    //pi_mod.lg_alpha = INVERSE R::trigamma(hcsigma*hcsigma);
    //pi_mod.lg_beta = exp(R::digamma(pi_mod.lg_alpha));
    pi_mod.a_drch = 1.0;
    std::vector<double> s_mod(p_mod, 0.);//pi_mod.s = new double [p_mod];
    for (size_t k=0; k<p_mod; ++k) {
        s_mod[k] = pi_mod.a_drch/p_mod;//s_mod.push_back(pi_mod.a_drch/p_mod);//pi_mod.s[k] = pi_mod.a_drch/p_mod;
        //s_mod.push_back(0.);
    }
    //s_mod[1] = 0.5;
    //s_mod[4] = 0.5;


    pinfo pi_con;
    pi_con.pbd = 0.7;//0.5;
    pi_con.pb = 0.35;//0.25;
    pi_con.pchange = 0.3;//0.4;
    pi_con.pswap = 0.0;//0.1;

    pi_con.alpha = con_alpha;
    pi_con.beta = con_beta;
    pi_con.tau = new double [p_m];
    pi_con.sigma = new double [p_m];
    for (size_t k=0; k<p_m; ++k) {
        pi_con.tau[k] = zeta_[k]/(4.0*sqrt(ntree_con));
        pi_con.sigma[k] = sigest_[k]/fabs(mscale);
    }
    //pi_con.tau = zeta/(4*sqrt(ntree_con)); //con_sd/(sqrt(delta_con)*sqrt((double) ntree_con));
    //pi_con.sigma = sigest/fabs(mscale);
    pi_con.a_drch = 1.0;
    std::vector<double> s_con(p_con,0.);//pi_con.s = new double [p_con];
    for (size_t k=0; k<p_con; ++k) {
        s_con[k] = pi_con.a_drch/p_con;//s_con.push_back(pi_con.a_drch/p_con);//pi_con.s[k] = pi_con.a_drch/p_con;
        //s_con.push_back(0.);
    }
    //s_con[2] = 0.5;
    //s_con[3] = 0.5;

    //double sigma = sigest;
    std::vector<double> sigma(p_m);
    for (size_t k=0; k<p_m; ++k) {
        sigma[k] = 2.5*sigest_[k];
    }
    
    
    arma::mat Covsigma(p_m, p_m, arma::fill::eye);
    
    
    pinfo pi_coxcon;
    pi_coxcon.pbd = 0.7;//0.5;
    pi_coxcon.pb = 0.35;//0.25;
    pi_coxcon.pchange = 0.3;//0.4;
    pi_coxcon.pswap = 0.0;//0.1;
    
    pi_coxcon.alpha = coxcon_alpha;
    pi_coxcon.beta = coxcon_beta;
    //pi_coxcon.tau = new double [?];
    //pi_coxcon.sigma = new doule [?];
    double hcsigma_2 = 0.25;//hcsig_coxcon_[0]; // gen.halfcauchy(1.5/sqrt((double)ntree_coxcon));//hcsig_coxcon_[0]; //
    pi_coxcon.lg_alpha = 1.0/(hcsigma_2*hcsigma_2) + 0.5;
    pi_coxcon.lg_beta = 1.0/(hcsigma_2*hcsigma_2);
    //pi_coxcon.lg_beta = exp(Rcpp::digamma(lg_alpha));
    pi_coxcon.a_drch = 1.0;
    std::vector<double> s_coxcon(p_coxcon, 0.);//pi_coxcon.s = new double [p_coxcon];
    for (size_t k=0; k<p_coxcon; ++k) {
        s_coxcon[k]=pi_coxcon.a_drch/p_coxcon;//s_coxcon.push_back(pi_coxcon.a_drch/p_coxcon);//pi_coxcon.s[k] = pi_coxcon.a_drch/p_coxcon;
    }

    std::vector<double> hintv;
    for (NumericVector::iterator it = intv_.begin(); it!=intv_.end(); ++it) {
        hintv.push_back(*it);
    }
    size_t G = hintv.size()-1;
    //Rcout << "number of pieces for baseline hzd: "
    //<< G << endl;
    std::vector<double> h(n), whichint(n), h0(G,1.0);
    for (size_t i = 0; i<n; ++i) {
        for (size_t j=0; j<G; ++j) {
            if ((yobs[i] > hintv[j]) && (yobs[i]<=hintv[j+1])) {
                h[i] = h0[j];
                whichint[i] = j;
            }
        }
    }
    
    //Rcout << "hintv: " << hintv << endl;
    //Rcout << "h0: " << h0 << endl;
    //Rcout << "h: " << h << endl;
    //Rcout << "whichint: " << whichint << endl;
    
    // To deal with baseline hazard through Pspline
    size_t degree = 3;
    size_t nseg = nknots - degree - 1;
    size_t ngrids = 20;
    
    std::vector<double> knots(nknots);
    //std::vector<double> gamma(nseg, 0.);
    double* gamma = new double[nseg];
    double* MHcounts_gamma = new double[nseg];
    double* hzd = new double[n];
    double* cumhzd = new double[n];
    double* cumhzd_survp = new double[ngrids + 1];
    //double sumgam = 0.;
    for (size_t k=0; k<nseg; ++k) {
        gamma[k] = 0.;
        //sumgam += gamma[k];
        MHcounts_gamma[k] = 0.;
    }
    //gamma[0] = gamma[0] - sumgam;
    
    for (int i = 0; i < nknots; ++i) {
            knots[i] = static_cast<double>(i) / (nknots - 1) * (maxy+0.001);
        }
    
    Rcout << "Knots for Bspline: " << knots << endl;
    
    //Initialize gamma
    /*
    for (size_t i=0; i<nseg; ++i) {
        gamma[i] = 1/nseg;
    }*/
    
    //Initialize the penalty parameter rho
    double rho = 0.0;
    
    //Initilize the (cumulative) baseline hazard for each individual
    for (size_t i=0; i<n; ++i) {
     /*   for (size_t j=0; j<m; ++j) {
            double basis = std::max(0., bspline(yobs[i], j, degree, knots));
            hzd[i] += gamma[j]*basis;
        }*/
        hzd[i] = GetHZD(yobs[i], degree, knots, gamma);
        cumhzd[i] = GetCumhzd(yobs[i], degree, ngrids, knots, gamma);
    }
    //Rcout << hzd << endl;
    //Rcout << cumhzd << endl;
    
    
    // for baseline hazard h0_k
    /*
    std::vector<double> y_fail;
    for (size_t k=0; k<n; ++k) {
        if (delta[k] >0) {
            y_fail.push_back(yobs[k]);
        }
    }
    std::set<double> s(y_fail.begin(), y_fail.end());
    std::vector<double> yord; yord.assign(s.begin(), s.end());
    size_t n_uniy = yord.size();
    size_t G = n_uniy;
    if (n_uniy != (n-num_censored)) {
        Rcout << "dim mismatch: n_uniyfailed (" << n_uniy <<") and n - num_censored (" << n - num_censored << ")" << endl;
    }
    std::vector<double> h0(n_uniy), h(n_uniy);
    h0[0]=yord[0]; h[0] = h0[0];//gen.gamma(h0[0],1/c_gp);
    for (size_t k=1; k<n_uniy; ++k) {
        h0[k] = yord[k] - yord[k-1];
        h[k]=h0[k];//gen.gamma(h0[k]/c_gp,1/c_gp); // c_gp for this gamma process prior.
    }
    //h = h0; // in case the initial h from gen.gamma is weird*/
    // Initialize dinfo
    double* allfit_con = new double[n*p_m]; // sum of fit of all trees
    //for (size_t i=0; i<n; i++) allfit_con[i] = ybar;
    for (size_t i=0; i<n; ++i) {
        for (size_t k=0; k<p_m; ++k) {
            allfit_con[i*p_m+k] = 2.5;//mbar[k];
        }
    }
    double* r_con = new double[n*p_m]; // residual for tree_con
    dinfo di_con;
    di_con.N = n;
    di_con.p = p_con;
    di_con.x = &x_con[0];
    di_con.y = r_con; //Note: y for each draw is the residual!! y - (allfit - ftemp[fit of current tree]) = y - allfit + ftemp
    di_con.p_y = p_m;
    di_con.delta = &delta[0];
    
    
    
    // dinfo for treatment effect funtion b(x) or tau(x), for both m and y equation;
    double* allfit_mod = new double[n*(p_m+1)];
    //for (size_t i=0; i<n; i++) allfit_mod[i] = (a_[i]*bscale1 + (1-a_[i])*bscale0)*trt_init;
    for (size_t i=0; i<n; ++i) {
        for (size_t k=0; k<p_m; ++k) {
            allfit_mod[i*(p_m+1)+k] = (a_[i]*bscale1 + (1-a_[i])*bscale0)*trt_init;
        }
        allfit_mod[i*(p_m+1)+p_m] = a_[i]*trt_init;
    }
    double* r_mod = new double[n*(p_m+1)]; // residual
    dinfo di_mod;
    di_mod.N = n;
    di_mod.p = p_mod;
    di_mod.x = &x_mod[0];
    di_mod.y = r_mod;
    di_mod.p_y = p_m+1;
    di_mod.delta = &deltaA[0];
    
    
    
    /*dinfo for the cox_con model */
    double* allfit_coxcon = new double[n];
    for (size_t i=0; i<n; i++) allfit_coxcon[i] = -1.;//ybar;
    double* r_coxcon = new double[n];
    dinfo di_coxcon;
    di_coxcon.N = n;
    di_coxcon.p = p_coxcon;
    di_coxcon.x = &x_coxcon[0];
    di_coxcon.y = r_coxcon;
    di_coxcon.p_y = 1;
    di_coxcon.delta = &delta[0];
    
    // for Covsigma prior
    arma::mat covsigprior(p_m, p_m, arma::fill::eye);
    for (size_t k=0; k<p_m; k++) {
        covsigprior(k,k) = covsigprior(k,k)*nu*kappa_[k];
    }
    
    
    // define the ProbHyper objects for mod, con, and coxcon ensembles;
    double PH_zeta = 1;
    double tmpsum = 0.;
    std::vector<double> logprior_mod(p_mod, 0.);
    for (size_t k=0; k<p_mod; ++k) {
        logprior_mod[k] = 1. / pow(k+1,PH_zeta);
        tmpsum += 1. / pow(k+1,PH_zeta);
    }
    for (size_t k=0; k<p_mod; ++k) {
        logprior_mod[k] = log(logprior_mod[k]/tmpsum);
    }
    
    ProbHypers PH_mod(pi_mod.a_drch, logprior_mod, false);
    /*
    Rcout << "PH_mod.d_mass " << PH_mod.dirichlet_mass << std::endl;
    Rcout << "PH_mod.log_mass " << PH_mod.log_mass << std::endl;
    Rcout << "PH_mod.counts " << PH_mod.counts << std::endl;
    Rcout << "PH_mod.use_c " << PH_mod.use_counts << std::endl;
    Rcout << "PH_mod.logp " << PH_mod.log_prior << std::endl;*/
    
    tmpsum = 0.;
    std::vector<double> logprior_con(p_con, 0.);
    for (size_t k=0; k<p_con; ++k) {
        logprior_con[k] = 1. / pow(k+1,PH_zeta);
        tmpsum += 1. / pow(k+1,PH_zeta);
    }
    for (size_t k=0; k<p_con; ++k) {
        logprior_con[k] = log(logprior_con[k]/tmpsum);
    }
    
    ProbHypers PH_con(pi_con.a_drch, logprior_con, false);
    
    tmpsum = 0.;
    std::vector<double> logprior_coxcon(p_coxcon, 0.);
    for (size_t k=0; k<p_coxcon; ++k) {
        logprior_coxcon[k] = 1. / pow(k+1,PH_zeta);
        tmpsum += 1. / pow(k+1,PH_zeta);
    }
    for (size_t k=0; k<p_coxcon; ++k) {
        logprior_coxcon[k] = log(logprior_coxcon[k]/tmpsum);
    }
    
    ProbHypers PH_coxcon(pi_con.a_drch, logprior_coxcon, false);
    
    
    // ------------------------------
    // store the fits
    double* allfit_m = new double[n*p_m]; //mhat
    for (size_t i=0; i<n; i++) {
        for (size_t k=0; k<p_m; ++k) {
            allfit_m[i*(p_m)+k] = allfit_mod[i*(p_m+1)+k] + allfit_con[i*(p_m)+k];
        }
    }
    
    double* allfit_y = new double[n];
    for (size_t i=0; i<n; i++) {
        allfit_y[i] = allfit_coxcon[i] + allfit_mod[i*(p_m+1)+p_m];
    }
    
    double* ftemp_m = new double[n*p_m]; // fit of current tree
    double* ftemp_y = new double[n];
    double* ftemp_mny = new double[n*(p_m+1)];
    // initialization ended
    
    NumericMatrix sigma_post(nd, p_m);
    NumericMatrix covsigma_post(nd, p_m*p_m);
    NumericVector msd_post(nd);
    NumericVector bsd_post(nd);
    NumericVector b0_post(nd);
    NumericVector b1_post(nd);
    NumericMatrix m_post(nd,n*p_m);// store allfit_con post
    NumericMatrix coxcon_post(nd,n);//store allfit_coxcon
    NumericMatrix coxtau_post(nd,n);//store allfit_coxcon
    NumericMatrix yhat_post(nd,n);
    NumericMatrix mhat_post(nd, n*p_m); //store mediator hat
    NumericMatrix b_post(nd,n*(p_m));// store allfit_mod po
    IntegerMatrix varcnt_con(nd,p_con);
    IntegerMatrix varcnt_mod(nd,p_mod);
    IntegerMatrix varcnt_coxcon(nd, p_coxcon);
    NumericMatrix hzd_post(nd, n);
    NumericMatrix scon_post(nd, p_con);
    NumericMatrix smod_post(nd, p_mod);
    NumericMatrix scoxcon_post(nd, p_coxcon);
    NumericMatrix adrch_post(nd, 3);
    NumericVector redundt(nd);
    NumericMatrix hzd_survp(nd, ngrids+1);
    
    arma::mat tree_counts_con(p_con, ntree_con, arma::fill::zeros);
    arma::mat tree_counts_coxcon(p_coxcon, ntree_coxcon, arma::fill::zeros);
    arma::mat tree_counts_mod(p_mod, ntree_mod, arma::fill::zeros);
    
    int save_tree_precision = 32;
    // save stuff to tree file;
    treef_con << std::setprecision(save_tree_precision) << xi_con <<endl;
    treef_con << ntree_con << endl;
    treef_con << di_con.p << endl;
    treef_con << di_con.p_y << endl;
    treef_con << nd << endl;

    treef_mod << std::setprecision(save_tree_precision) << xi_mod << endl;
    treef_mod << ntree_mod << endl;
    treef_mod << di_mod.p << endl;
    treef_mod << di_mod.p_y << endl;
    treef_mod << nd << endl;
                       
    treef_coxcon << std::setprecision(save_tree_precision) << xi_coxcon <<endl;
    treef_coxcon << ntree_coxcon << endl;
    treef_coxcon << di_coxcon.p << endl;
    treef_coxcon << di_coxcon.p_y << endl;
    treef_coxcon << nd << endl;
    
    /* ------------ MCMC ----------------*/

    Rcout << "\n====================================\nBeginning MCMC:\n====================================\n";
    time_t tp;
    int time1 = time(&tp);

    size_t save_ctr = 0;
    bool verbose_itr = false;

    double* weight_m = new double[n*p_m];
    double* weight_heter = new double[n*(p_m+1)];
    double* weight_y = new double[n];
    //double* weight_y_heter = new double[n];

    logger.setLevel(0);

    bool printTrees = false; double u, u_MHp = 0.0; bool varselect = false; double tmpbar = 0.;
    
    for (size_t iIter=0; iIter<(nd*thin+burn); iIter++) {
        //verbose_itr = true;
        verbose_itr = false;//iIter>=burn;
        printTrees = false;//iIter>=burn;
        verbose_sigma = true;
        if (vs) {
            varselect = (iIter>=(burn/2));
        }
        PH_mod.use_counts = varselect; PH_con.use_counts = varselect; PH_coxcon.use_counts = varselect;

        if (verbose_sigma) {
            if (iIter%printevery==0) {
                Rcout << "iteration: " << iIter << " sigma: " << sigma << endl;
            }
        }

        logger.setLevel(verbose_itr);
        
        logger.log("=========================================");
        sprintf(logBuff, "MCMC iteration: %d of %d Start", iIter+1, nd*thin+burn);
        logger.log(logBuff);
        sprintf(logBuff, "sigma %f, %f, ...", sigma[0], sigma[1]);
        logger.log(logBuff);
        logger.log("==========================================");
        
        if (verbose_itr) {
            logger.getVectorHead(y, logBuff);
            Rcout << "            y: " << logBuff << "\n";

            logger.getVectorHead(allfit_y, logBuff);
            Rcout << "Current Fit hzd : " << logBuff << "\n";
            
            logger.getVectorHead(m, logBuff);
            Rcout << "            m : " << logBuff << "\n";
            
            logger.getVectorHead(allfit_m, logBuff);
            Rcout << "Current Fit m : " << logBuff << "\n";

            logger.getVectorHead(allfit_con, logBuff);
            Rcout << "allfit_con : " << logBuff << "\n";

            logger.getVectorHead(allfit_mod, logBuff);
            Rcout << "allfit_mod : " << logBuff << "\n";
            
            logger.getVectorHead(allfit_coxcon, logBuff);
            Rcout << "allfit_coxcon : " << logBuff << "\n";
        }
        
        for (size_t k=0; k<n; ++k) {
            for (size_t l=0; l<p_m; ++l) {
                weight_m[(k*p_m)+l] = w[k]*mscale*mscale/(sigma[l]*sigma[l]);
            }
            weight_y[k] = w[k];
        }
        
        //riskset(y, h, yord, delta, weight_y);
        riskset(y, h0, hintv, whichint, weight_y);
        // Rcout << "weight_y : ";
        for(size_t k=0; k<n; ++k) {
            weight_y[k] = cumhzd[k];
            //Rcout << weight_y[k] << " ";
        }
         
    
        for (size_t k=0; k<ntrt; ++k) {
            for (size_t j=0; j<p_m; ++j) {
                weight_heter[k*(p_m+1)+j] = w[k]*bscale1*bscale1/(sigma[j]*sigma[j]);
            }
            weight_heter[k*(p_m+1)+p_m] = weight_y[k];
        }
        for (size_t k=ntrt; k<n; ++k) {
            for (size_t j=0; j<p_m; ++j) {
                weight_heter[k*(p_m+1)+j] = w[k]*bscale0*bscale0/(sigma[j]*sigma[j]);
            }
            weight_heter[k*(p_m+1)+p_m] = 0.0*weight_y[k];
        }
      
        logger.log("==========================");
        logger.log("-- Tree Processing --");
        logger.log("==========================");
        
        
        
        
        // draw trees for m(x), the mediator control tree;
        for (size_t iTreecon=0; iTreecon<ntree_con; iTreecon++) {
            
            logger.log("===========================");
            sprintf(logBuff, "Updating Control Tree: %d of %d", iTreecon+1, ntree_con);
            logger.log(logBuff);
            logger.log("===========================");
            logger.startContext();
            
            logger.log("Attempting to Print MedCon Tree pre Update \n");
            if (verbose_itr && printTrees) {
                t_con[iTreecon].pr(xi_con);
                Rcout << "\n\n";
            }
            
            //Rcout << t_con[iTreecon].getmu()[0] << " " << t_con[iTreecon].getmu()[1] << endl;
            
            fit(t_con[iTreecon], xi_con, di_con, ftemp_m);
            /*for (size_t k=0; k<5; ++k) {
                Rcout << ftemp_m[k] << " ";
            }
            Rcout << endl;*/
            
            logger.log("Attempting to Print MedCon Tree Post first call to fit \n");
            if (verbose_itr && printTrees) {
                t_con[iTreecon].pr(xi_con);
                Rcout << "\n\n";
            }
            
            for (size_t k=0; k<(n*p_m); k++) {
                if (ftemp_m[k] != ftemp_m[k]) {
                    Rcout << "control tree " << iTreecon << " obs " << k << " " << endl;
                    Rcout << t_con[iTreecon] << endl;
                    stop("nan in ftemp");
                }
            }
            
            for (size_t k=0; k<(n*p_m); k++) {
                allfit_m[k] = allfit_m[k] - mscale*ftemp_m[k];
                allfit_con[k] = allfit_con[k] - mscale*ftemp_m[k];
                r_con[k] = (m[k]-allfit_m[k])/mscale;
                if (r_con[k] != r_con[k]) {
                    Rcout << (m[k]-allfit_m[k]) << endl;
                    Rcout << mscale << endl;
                    Rcout << r_con[k] << endl;
                    stop("NaN in resid_con");
                }
            }
            
            if (verbose_itr && printTrees) {
                logger.getVectorHead(weight_m, logBuff);
                Rcout << "\n weight: " << logBuff << "\n\n";
            }
            
            u_MHp = gen.uniform();
            if (u_MHp<0.4) {
                logger.log("Starting MH-prior-step Processing");
                logger.startContext();
                draw_prior_withvs(t_con[iTreecon], xi_con, di_con, weight_m, pi_con, gen, logger, nv_con, Covsigma, covsigprior, varselect, PH_con);
                logger.stopContext();

                logger.log("Attempting to Print MedCon Tree Post MH-prior-step \n");
                if (verbose_itr && printTrees) {
                    t_con[iTreecon].pr(xi_con);
                    Rcout << "\n";
                }
            }
            
            
            u = gen.uniform();
            //Rcout << u << endl;
            if (u<pi_con.pbd) {
                logger.log("Starting Birth/Death Processing");
                logger.startContext();
                bd_withvs(t_con[iTreecon], xi_con, di_con, weight_m, pi_con, gen, logger, nv_con, Covsigma, covsigprior, varselect, s_con, PH_con);
                logger.stopContext();

                logger.log("Attempting to Print MedCon Tree Post bd \n");
                if (verbose_itr && printTrees) {
                    t_con[iTreecon].pr(xi_con);
                    Rcout << "\n";
                }
            } else if(u < (pi_con.pswap + pi_con.pbd)) {
                logger.log("Starting SwapRule Processing");
                logger.startContext();
                swaprule(t_con[iTreecon], xi_con, di_con, weight_m, pi_con, gen, logger, Covsigma, covsigprior);
                logger.stopContext();

                logger.log("Attempting to Print Tree Post SwapRule \n");
                if (verbose_itr && printTrees) {
                    t_con[iTreecon].pr(xi_con);
                    Rcout << "\n";
                }
            } else {
                logger.log("Starting ChangeRule Processing");
                logger.startContext();
                changerule_withvs(t_con[iTreecon], xi_con, di_con, weight_m, pi_con, gen, logger, nv_con, Covsigma, covsigprior, varselect, s_con, PH_con);
                logger.stopContext();

                logger.log("Attempting to Print Tree Post ChangeRule \n");
                if (verbose_itr && printTrees) {
                    t_con[iTreecon].pr(xi_con);
                    Rcout << "\n";
                }
            }
            
            if (verbose_itr && printTrees) {
                logger.log("Printing Current Status of Fit");

                logger.getVectorHead(a_, logBuff);
                Rcout << "\n           a : " << logBuff << "\n";
                
                logger.getVectorHead(m, logBuff);
                Rcout << "\n           m : " << logBuff << "\n";

                logger.getVectorHead(allfit_m, logBuff);
                Rcout << "Current Fit_M : " << logBuff << "\n";

                logger.getVectorHead(r_con, logBuff);
                Rcout << "      r_con : " << logBuff << "\n\n";
            }
            
            
            
            
            
            logger.log("Strarting to draw mu");
            logger.startContext();

            drmu(t_con[iTreecon], xi_con, di_con, pi_con, weight_m, gen, Covsigma, covsigprior);

            logger.stopContext();
            
            logger.log("Attempting to Print Tree Post drmu \n") ;
            if (verbose_itr && printTrees) {
                t_con[iTreecon].pr(xi_con);
                Rcout << "\n";
            }

            
            fit(t_con[iTreecon], xi_con, di_con, ftemp_m);
            
            for (size_t k=0; k<(n*p_m); k++) {
                allfit_m[k] += mscale*ftemp_m[k];
                allfit_con[k] += mscale*ftemp_m[k];
            }

            logger.log("Attempting to Print Tree Post second call to fit \n");
            
            if (verbose_itr && printTrees) {
                t_con[iTreecon].pr(xi_con);
                Rcout << "\n";
            }
            
            logger.stopContext();
        } // end con_tree loop
        
        std::vector<double> store4hcsig21;
        std::vector<double> store4hcsig22;
        std::vector<double> storemu2;
        
        
        // Next, for coxcon trees in the PH model;
        for (size_t iTreecoxcon=0; iTreecoxcon<ntree_coxcon; iTreecoxcon++) {
            
            logger.log("===========================");
            sprintf(logBuff, "Updating Cox_Control Tree: %d of %d", iTreecoxcon+1, ntree_coxcon);
            logger.log(logBuff);
            logger.log("===========================");
            logger.startContext();
            
            logger.log("Attempting to Print Tree pre Update \n");
            if (verbose_itr && printTrees) {
                t_coxcon[iTreecoxcon].pr(xi_coxcon);
                Rcout << "\n\n";
            }
            
            fit(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, ftemp_y);
            
            /*for (size_t k=0; k<10; ++k) {
                Rcout << ftemp_y[k] << " ";
            }
            Rcout << endl;*/
            
            logger.log("Attempting to Print CoxCon Tree Post first call to fit \n");
            if (verbose_itr && printTrees) {
                t_coxcon[iTreecoxcon].pr(xi_coxcon);
                Rcout << "\n\n";
            }
            
            for (size_t k=0; k<n; k++) {
                if (ftemp_y[k] != ftemp_y[k]) {
                    Rcout << "Coxcontrol tree " << iTreecoxcon << " obs " << k << " " << endl;
                    Rcout << t_coxcon[iTreecoxcon] << endl;
                    stop("nan in ftemp");
                }
            }
            
            for (size_t k=0; k<n; k++) {
                allfit_y[k] = allfit_y[k] - ftemp_y[k];
                allfit_coxcon[k] = allfit_coxcon[k] - ftemp_y[k];
                r_coxcon[k] = exp(allfit_y[k]);
                if (r_coxcon[k] != r_coxcon[k]) {
                    Rcout << allfit_y[k] << endl;
                    Rcout << ftemp_y[k] << endl;
                    Rcout << r_coxcon[k] << endl;
                    stop("NaN in resid_coxcon");
                }
            }
            
            if (verbose_itr && printTrees) {
                logger.getVectorHead(weight_y, logBuff);
                Rcout << "\n weight: " << logBuff << "\n\n";
            }
            
          /*  Rcout << "\n weight: " << endl;
            double sum_wy = 0.0;
            for(size_t k=0; k<n; ++k){
                Rcout << weight_y[k] << " ";
                sum_wy += weight_y[k];
            }
            Rcout << "\nsum: " << sum_wy << endl;*/
            
            
            //Rcout << "lg_alpha: " << pi_coxcon.lg_alpha << ", lg_beta: " << pi_coxcon.lg_beta << endl;
            
            u_MHp = gen.uniform();
            if (u_MHp<0.4) {
                logger.log("Starting MH-prior-step Processing");
                logger.startContext();
                draw_prior_lgwithvs(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, weight_y, pi_coxcon, gen, logger, nv_coxcon, Covsigma, covsigprior, varselect, PH_coxcon);
                logger.stopContext();

                logger.log("Attempting to Print CoxCon Tree Post MH-prior-step \n");
                if (verbose_itr && printTrees) {
                    t_coxcon[iTreecoxcon].pr(xi_coxcon);
                    Rcout << "\n";
                }
            }
            
            u = gen.uniform();
            //Rcout << u << endl;
            if (u<pi_coxcon.pbd) {
                logger.log("Starting Birth/Death Processing");
                logger.startContext();
                bd_lgwithvs(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, weight_y, pi_coxcon, gen, logger, nv_coxcon, Covsigma, covsigprior, varselect, s_coxcon, PH_coxcon);
                logger.stopContext();

                logger.log("Attempting to Print CoxCon Tree Post bd \n");
                if (verbose_itr && printTrees) {
                    t_coxcon[iTreecoxcon].pr(xi_coxcon);
                    Rcout << "\n";
                }
            } else if(u < (pi_coxcon.pswap + pi_coxcon.pbd)) {
                logger.log("Starting SwapRule Processing");
                logger.startContext();
                swaprule_lg(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, weight_y, pi_coxcon, gen, logger, Covsigma, covsigprior);
                logger.stopContext();

                logger.log("Attempting to Print CoxCon Tree Post SwapRule \n");
                if (verbose_itr && printTrees) {
                    t_coxcon[iTreecoxcon].pr(xi_coxcon);
                    Rcout << "\n";
                }
            } else {
                logger.log("Starting ChangeRule Processing");
                logger.startContext();
                changerule_lgwithvs(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, weight_y, pi_coxcon, gen, logger, nv_coxcon, Covsigma, covsigprior, varselect, s_coxcon, PH_coxcon);
                logger.stopContext();

                logger.log("Attempting to Print Coxcon Tree Post ChangeRule \n");
                if (verbose_itr && printTrees) {
                    t_coxcon[iTreecoxcon].pr(xi_coxcon);
                    Rcout << "\n";
                }
            }
            
            
            
            if (verbose_itr && printTrees) {
                logger.log("Printing Current Status of Fit");

                logger.getVectorHead(a_, logBuff);
                Rcout << "\n           a : " << logBuff << "\n";
                
                logger.getVectorHead(y, logBuff);
                Rcout << "\n           y : " << logBuff << "\n";
                
                logger.getVectorHead(truehzd_, logBuff);
                Rcout << "\n           hzd : " << logBuff << "\n";

                logger.getVectorHead(allfit_y, logBuff);
                Rcout << "Current Fit hzd: " << logBuff << "\n";

                logger.getVectorHead(r_coxcon, logBuff);
                Rcout << "      r_coxcon : " << logBuff << "\n\n";
            }
            
            
            logger.log("Strarting to draw mu_lg");
            logger.startContext();

            drmu_withlg(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, pi_coxcon, weight_y, gen, Covsigma, covsigprior, store4hcsig21, store4hcsig22, storemu2);

            logger.stopContext();
            
            logger.log("Attempting to Print Coxcon Tree Post drmu_withlg \n") ;
            if (verbose_itr && printTrees) {
                t_coxcon[iTreecoxcon].pr(xi_coxcon);
                Rcout << "\n";
            }
            
            fit(t_coxcon[iTreecoxcon], xi_coxcon, di_coxcon, ftemp_y);
            
        
            for (size_t k=0; k<n; k++) {
                allfit_y[k] += ftemp_y[k];
                allfit_coxcon[k] += ftemp_y[k];
            }
            
            
            tmpbar = 0.;
            
            for (size_t i=0; i<n; ++i) {
                tmpbar += allfit_coxcon[i];
            }
            
            //for (size_t i=0; i<n; ++i) {
                //allfit_coxcon[i] = allfit_coxcon[i] - tmpbar/n;
                //allfit_y[i] = allfit_y[i] - tmpbar/n;
            //}
            
            logger.log("Attempting to Print CoxCon Tree Post second call to fit \n");

            if (verbose_itr && printTrees) {
                t_coxcon[iTreecoxcon].pr(xi_coxcon);
                Rcout << "\n";
            }
            
            
            logger.stopContext();
        } // end coxcon_tree loop
        
        
        // Next, for shared b(x) trees
        std::vector<double> store4hcsig11;
        std::vector<double> store4hcsig12;
        std::vector<double> storemu1;
        
        for (size_t iTreeMod=0; iTreeMod<ntree_mod; iTreeMod++) {
            logger.log("============================");
            sprintf(logBuff,"Updating Moderate Tree: %d of %d", iTreeMod+1, ntree_mod);
            logger.log(logBuff);
            logger.log("============================");
            logger.startContext();
            
            logger.log("Attempting to Print Mod Tree Pre Update \n");
            if(verbose_itr && printTrees){
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                  }
            
            fit(t_mod[iTreeMod],
                      xi_mod,
                      di_mod,
                      ftemp_mny);
            
          /*  for (size_t k=0; k<12; ++k) {
                Rcout << ftemp_mny[k] << " ";
            }
            Rcout << endl;*/

            logger.log("Attempting to Print Mod Tree Post first call to fit");
            if(verbose_itr && printTrees){
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                  }
            
            for (size_t k=0; k<(n*(p_m+1)); k++) {
                if (ftemp_mny[k] != ftemp_mny[k]) {
                    Rcout << "moderate tree " << iTreeMod << " obs " << k << " " << endl;
                    Rcout << t_mod[iTreeMod] << endl;
                    stop("nan in ftemp");
                }
            }
            
            
            for (size_t k=0; k<n; k++) {
                for (size_t l=0; l<p_m; ++l) {
                    double bscale = (k<ntrt) ? bscale1 : bscale0;
                    allfit_m[k*p_m+l] = allfit_m[k*p_m+l] - bscale*ftemp_mny[k*(p_m+1)+l];
                    allfit_mod[k*(p_m+1)+l] = allfit_mod[k*(p_m+1)+l] - bscale*ftemp_mny[k*(p_m+1)+l];
                    r_mod[k*(p_m+1)+l] = (m[k*p_m+l] - allfit_m[k*p_m+l])/bscale;
                }
                allfit_y[k] = allfit_y[k] - a[k]*ftemp_mny[k*(p_m+1)+p_m];
                allfit_mod[k*(p_m+1)+p_m] = allfit_mod[k*(p_m+1)+p_m] - a[k]*ftemp_mny[k*(p_m+1)+p_m];
                r_mod[k*(p_m+1)+p_m] = exp(allfit_y[k]);
            }
            
            u_MHp = gen.uniform();
            if (u_MHp<0.4) {
                logger.log("Starting MH-prior-step Processing");
                logger.startContext();
                draw_prior_lgwithvs(t_mod[iTreeMod], xi_mod, di_mod, weight_heter, pi_mod, gen, logger, nv_mod, Covsigma, covsigprior, varselect, PH_mod);
                logger.stopContext();

                logger.log("Attempting to Print Mod Tree Post MH-prior-step \n");
                if (verbose_itr && printTrees) {
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                }
            }
            
            u = gen.uniform();
            //Rcout << u << endl;
            if (u<pi_mod.pbd) {
                logger.log("Starting Birth/Death Processing");
                logger.startContext();
                bd_lgwithvs(t_mod[iTreeMod], xi_mod, di_mod, weight_heter, pi_mod, gen, logger, nv_mod, Covsigma, covsigprior, varselect, s_mod, PH_mod);
                logger.stopContext();

                logger.log("Attempting to Print Mod Tree Post bd \n");
                if (verbose_itr && printTrees) {
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                }
            } else if (u < (pi_mod.pbd + pi_mod.pswap)) {
                logger.log("Starting SwapRule Processing");
                logger.startContext();
                swaprule_lg(t_mod[iTreeMod], xi_mod, di_mod, weight_heter, pi_mod, gen, logger, Covsigma, covsigprior);
                logger.stopContext();

                logger.log("Attempting to Print Mod Tree Post SwapRule \n");
                if (verbose_itr && printTrees) {
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                }
            } else {
                logger.log("Starting ChangeRule Processing");
                logger.startContext();
                changerule_lgwithvs(t_mod[iTreeMod], xi_mod, di_mod, weight_heter, pi_mod, gen, logger, nv_mod, Covsigma, covsigprior, varselect, s_mod, PH_mod);
                logger.stopContext();

                logger.log("Attempting to Print Mod Tree Post ChangeRule \n");
                if (verbose_itr && printTrees) {
                    t_mod[iTreeMod].pr(xi_mod);
                    Rcout << "\n";
                }
            }
            
            
            
            if (verbose_itr && printTrees) {
                logger.log("Printing Status of Fit");

                logger.getVectorHead(a_, logBuff);
                Rcout << "\n       a : " << logBuff << "\n";
                
                logger.getVectorHead(m, logBuff);
                Rcout << "         m : " << logBuff << "\n";
                
                logger.getVectorHead(truehzd_, logBuff);
                Rcout << "         hzd : " << logBuff << "\n";

                logger.getVectorHead(allfit_m, logBuff);
                Rcout << " Fit_M - Tree : " << logBuff << "\n";
                
                logger.getVectorHead(allfit_y, logBuff);
                Rcout << " Fit_hzd - Tree : " << logBuff << "\n";
                
                logger.getVectorHead(r_mod, logBuff);
                Rcout << "    r_mod : " << logBuff << "\n\n";
                
                Rcout << "mscale: " << mscale << "\n";

                Rcout << "bscale0: " << bscale0 << "\n";

                Rcout << "bscale1: " << bscale1 << "\n\n";
            }
            
            logger.log("Starting to draw mu and mu_lg");
            logger.startContext();
            drmu_withlg(t_mod[iTreeMod], xi_mod, di_mod, pi_mod, weight_heter, gen, Covsigma, covsigprior, store4hcsig11, store4hcsig12, storemu1);
            logger.stopContext();
            
            logger.log("Attempting to Print Mod Tree Post drmuheter \n");
            if (verbose_itr && printTrees) {
                t_mod[iTreeMod].pr(xi_mod);
                Rcout << "\n";
            }
            
            fit(t_mod[iTreeMod], xi_mod, di_mod, ftemp_mny);
            
            for (size_t k=0; k<ntrt; k++) {
                for (size_t l=0; l<p_m; ++l) {
                    allfit_m[k*p_m+l] += bscale1*ftemp_mny[k*(p_m+1)+l];
                    allfit_mod[k*(p_m+1)+l] += bscale1*ftemp_mny[k*(p_m+1)+l];
                }
                allfit_y[k] += a[k]*ftemp_mny[k*(p_m+1)+p_m];
                allfit_mod[k*(p_m+1)+p_m] += a[k]*ftemp_mny[k*(p_m+1)+p_m];
            }
            
            for (size_t k=ntrt; k<n; k++) {
                for (size_t l=0; l<p_m; ++l) {
                    allfit_m[k*p_m+l] += bscale0*ftemp_mny[k*(p_m+1)+l];
                    allfit_mod[k*(p_m+1)+l] += bscale0*ftemp_mny[k*(p_m+1)+l];
                }
                allfit_y[k] += a[k]*ftemp_mny[k*(p_m+1)+p_m];
                allfit_mod[k*(p_m+1)+p_m] += a[k]*ftemp_mny[k*(p_m+1)+p_m];
            }
            
            logger.log("Attempting to Print Mod Tree Post second call to fit");

            if (verbose_itr && printTrees) {
                t_mod[iTreeMod].pr(xi_mod);
                Rcout << "\n";
            }
            
            logger.stopContext();
        } // end mod_tree loop
        
        logger.setLevel(verbose_itr);

        logger.log("============================");
        logger.log("-- MCMC iteration cleanup --");
        logger.log("============================");
        
        //-----------try bscale
       /* if(1) {
              double ww0 = 0.0, ww1 = 0.;
              double rw0 = 0.0, rw1 = 0.;
              //double s2 = sigma*sigma;
            std::vector<double> s2 (p_m);
            for(size_t k=0; k<p_m; ++k){
                s2[k] = sigma[k]*sigma[k];
            }
              for(size_t k=0; k<n; ++k) {
                double bscale = (k<ntrt) ? bscale1 : bscale0;
                //double scale_factor = (w[k]*allfit_mod[k]*allfit_mod[k])/(s2*bscale*bscale);
                  std::vector<double> scale_factor (p_m);
                  for(size_t l=0; l<p_m; ++l) {
                      scale_factor[l] = (w[k]*allfit_mod[k*p_m+l]*allfit_mod[k*p_m+l])/(s2[l]*bscale*bscale);
                  }

                if(scale_factor!=scale_factor) {
                  Rcout << " scale_factor " << scale_factor << endl;
                  stop("");
                }

                //double randeff_contrib = randeff ? allfit_random[k] : 0.0;

                //double r = (y[k] - allfit_con[k] - randeff_contrib)*bscale/allfit_mod[k];
                  std::vector<double> r (p_m);
                  for (size_t l=0; l<p_m; ++l) {
                      r[l] = (m[k*p_m+l] - allfit_con[k*p_m+l])*bscale/allfit_mod[k*p_m+l];
                  }

                if(r!=r) {
                  Rcout << "bscale " << k << " r " << r << " mscale " <<mscale<< " bscale " << bscale0 << " " <<bscale1 << endl;
                  stop("");
                }
                if(k<ntrt) {
                    for(size_t l=0; l<p_m; ++l) {
                        ww1 += scale_factor[l];
                        rw1 += r[l]*scale_factor[l];
                    }
                } else {
                    for(size_t l=0; l<p_m; ++l) {
                        ww0 += scale_factor[l];
                        rw0 += r[l]*scale_factor[l];
                    }
                }
              }
              logger.log("Drawing bscale 1");
              logger.startContext();
              double bscale1_old = bscale1;
              double bscale_fc_var = 1/(ww1 + 2.0);
              bscale1 = bscale_fc_var*rw1 + gen.normal(0., 1.)*sqrt(bscale_fc_var);
              if(verbose_itr){

                Rcout << "Original bscale1 : " << bscale1_old << "\n";
                Rcout << "bscale_prec : " << 2. << ", ww1 : " << ww1 << ", rw1 : " << rw1 << "\n";
                Rcout << "New  bscale1 : " << bscale1 << "\n\n";
              }
              logger.stopContext();


              logger.log("Drawing bscale 0");
              logger.startContext();
              double bscale0_old = bscale0;
              bscale_fc_var = 1/(ww0 + 2.);
              bscale0 = bscale_fc_var*rw0 + gen.normal(0., 1.)*sqrt(bscale_fc_var);
              if(verbose_itr){
                Rcout << "Original bscale0 : " << bscale0_old << "\n";
                Rcout << "bscale_prec : " << 2. << ", ww0 : " << ww0 << ", rw0 : " << rw0 << "\n";
                Rcout << "New  bscale0 : " << bscale0 << "\n\n";
              }
              logger.stopContext();

              for(size_t k=0; k<ntrt; ++k) {
                  for(size_t l=0; l<p_m; ++l) {
                      allfit_mod[k*p_m+l] = allfit_mod[k*p_m+l]*bscale1/bscale1_old;
                  }
              }
              for(size_t k=ntrt; k<n; ++k) {
                  for(size_t l=0; l<p_m; ++l) {
                      allfit_mod[k*p_m+l] = allfit_mod[k*p_m+l]*bscale0/bscale0_old;
                  }
                
              }
        }*/

        
        // -----------------
        logger.log("Draw Sigma");
        arma::mat mvecsum(p_m, p_m, arma::fill::zeros);
        for (size_t j=0; j<n; j++) {
            arma::mat mvectemp(p_m, 1, arma::fill::zeros);
            for (size_t k=0; k<p_m; ++k) {
                mvectemp(k,0) = m[j*(p_m)+k] - allfit_m[j*p_m+k];
            }
            mvecsum = mvecsum + w[j]*mvectemp*mvectemp.t();
        }
        
        Covsigma = gen.wishart((mvecsum + covsigprior).i(), nu+n);
        Covsigma = Covsigma.i();
        
        // ------------------
        /*
        for (size_t k=0; k<p_m; ++k) {
            double rss = 0.0;
            double restemp = 0.0;
            for (size_t j=0; j<n; j++) {
                restemp = m[j*(p_m)+k] - allfit_m[j*p_m+k];
                rss += w[j]*restemp*restemp;
            }
            sigma[k] = sqrt((nu*kappa_[k] + rss)/gen.chisq(nu+n));
            pi_con.sigma[k] = sigma[k]/fabs(mscale);
            pi_mod.sigma[k] = sigma[k];
        }*/
        
        for (size_t k=0; k<p_m; ++k) {
            sigma[k] = sqrt(Covsigma(k,k));
            pi_con.sigma[k] = sigma[k]/fabs(mscale);
            pi_mod.sigma[k] = sigma[k];
            
        }
        
        /*
        // -----------------
        logger.log("Draw hazard"); //note that we assume no ties for now
        double par2 = 0.1;
        for (size_t k=0; k<n_uniy; ++k) {
            for (size_t i=0; i<n; ++i) {
                if (yobs[i] >= yord[k]) {
                    par2 += exp(allfit_y[i]);
                }
            }
            //Rcout << "for h[" << k << "]" << endl;
            //Rcout << "par2: " << par2 << endl;
            h[k] = gen.gamma(1, 1/par2);//gen.gamma(h0[k]/c_gp+par1,1/c_gp + 1/par2);
            par2 = 0.1;
            //h[k] = 1.;
        }*/
        // ------------------
        //Rcout << h << endl;
        
        
        
        
        /*
        logger.log("Draw hazard"); //note that we assume no ties for now
        double par1 = 0.1, par2 = 0.2;
        for (size_t k=0; k<G; ++k) {
            for (size_t i=0; i<n; ++i) {
                if (whichint[i] == k) {
                    par1 += delta[i];
                    par2 += exp(allfit_y[i])*(yobs[i] - hintv[k]);
                } else if (whichint[i] > k) {
                    par2 += exp(allfit_y[i])*(hintv[k+1] - hintv[k]);
                }
            }
        //    Rcout << "for h0[" << k << "]" << endl;
        //    Rcout << "par1: " << par1 << endl;
        //    Rcout << "par2: " << par2 << endl;
            h0[k] = gen.gamma(par1,1/par2);//gen.gamma(h0[k]/c_gp+par1,1/c_gp + 1/par2);
            //h0[k] = h0[k]*exp(tmpbar);
            //h0[k] = h0[k]*exp(tmpbar);
            par1 = 0.1;
            par2 = 0.2;
            //h0[k] = 1.;
        }
        // ------------------
        //Rcout << h0 << endl;*/
        
        tmpbar = 0;
        for (size_t i=0; i<n; ++i) {
            tmpbar += allfit_coxcon[i];
        }
        /*for (size_t i=0; i<n; ++i) {
            allfit_coxcon[i] -= tmpbar/n;
            allfit_y[i] -= tmpbar/n;
        }*/
        
        
        //MH steps to update gamma
        logger.log("Draw Spline Coefs");
        
        //gammaMHcons(rho, 0.0001, degree, ngrids, 1.6, MHcounts_gamma, delta, yobs, allfit_y, knots, gamma, gen);
        gammaMH(rho, 0.0001, degree, ngrids, 0.03, MHcounts_gamma, delta, yobs, allfit_y, knots, gamma, gen);
        //for (size_t k=0; k<nseg; ++k) {
        //    gamma[k] = 0.;
        //}
        /*
        double gamsum = 0.;
        for (size_t l=0; l<nseg; ++l) {
            gamsum += gamma[l];
        }
        
        for (size_t l=0; l<nseg; ++l) {
            gamma[l] -= gamsum/nseg;
        }*/
        
        
        
        //update rho
        double pntysum = 0;
        pntysum += 0.0001*gamma[0]*gamma[0];
        pntysum += 0.0001*gamma[nseg]*gamma[nseg];
        for (size_t k=1; k < nseg-1; ++k) {
            pntysum += pow(gamma[k-1] -2*gamma[k] + gamma[k+1], 2);
            pntysum += 0.0001*gamma[k]*gamma[k];
        }
        //Rcout << "shape par for spline penalty: " << 1 + (nseg)/2 << "rate par for spline penalty: " << (0.001 + pntysum/2) << endl;
        rho = gen.gamma(1 + (nseg)/2, 1 / (0.001 + pntysum/2));
        
        
        logger.log("Update (cum) baseline hazard");
        //update cumhzd
        for (size_t i=0; i<n; ++i) {
            hzd[i] = GetHZD(yobs[i], degree, knots, gamma);
            cumhzd[i] = GetCumhzd(yobs[i], degree, ngrids, knots, gamma);
        }
        
        cumhzd_survp[0] = GetCumhzd(t_survp, degree, ngrids, knots, gamma);
        for (size_t i=1; i<(ngrids + 1); ++i) {
            cumhzd_survp[i] = GetCumhzd((i-1)*maxy/(ngrids-1), degree, ngrids, knots, gamma);
        }
        
       
        NumericVector params1 = NumericVector::create(0.0, 0.5/sqrt((double)ntree_mod));
        NumericVector params2 = NumericVector::create(0.0, 1.5/sqrt((double)ntree_coxcon));
        NumericVector hcvec1 = NumericVector::create(hcsigma_1);
        NumericVector hcvec2 = NumericVector::create(hcsigma_2);
        //double steps = 20;
        NumericVector v1 = hcsig_update(hcvec1, params1, store4hcsig11, store4hcsig12, storemu1, gen);//slice_sample_cpp(post_hcauchy, params1, hcvec1, store4hcsig11, store4hcsig12, storemu1, 1, 1, 0.001, 10);
        NumericVector v2 = hcsig_update(hcvec2, params2, store4hcsig21, store4hcsig22, storemu2, gen);//slice_sample_cpp(post_hcauchy, params2, hcvec2, store4hcsig21, store4hcsig22, storemu2, 1, 1, 0.001, 10);
        hcsigma_2 = v2[0];
        hcsigma_1 = v1[0];
     
       //hcsigma_2 = gen.halfcauchy(1.5/sqrt((double)ntree_coxcon));//hcsig_coxcon_[iIter+1]; //
       pi_coxcon.lg_alpha = 1.0/(hcsigma_2*hcsigma_2) + 0.5;//25.5;//
       pi_coxcon.lg_beta = 1.0/(hcsigma_2*hcsigma_2);//25;//
           
       //hcsigma_1 = gen.halfcauchy(0.5/sqrt((double)ntree_mod));//hcsig_mod_[iIter+1]; //
       pi_mod.lg_alpha = 1.0/(hcsigma_1*hcsigma_1) + 0.5;//15.5;//
       pi_mod.lg_beta = 1.0/(hcsigma_1*hcsigma_1);//15;//
        
        
        //Update S
        /*
        if (varselect) {
            UpdateS(di_con, pi_con, gen, nv_con, s_con);
            UpdateS(di_coxcon, pi_coxcon, gen, nv_coxcon, s_coxcon);
            UpdateS(di_mod, pi_mod, gen, nv_mod, s_mod);
        }*/
        
        
        // store posterior draws
        if (((iIter>=burn) & (iIter % thin==0))) {
            for (size_t j=0; j<ntree_con; j++) treef_con << std::setprecision(save_tree_precision) << t_con[j] << endl; // save trees
            for (size_t j=0; j<ntree_mod; j++) treef_mod << std::setprecision(save_tree_precision) << t_mod[j] << endl;
            
            for (size_t j=0; j<ntree_coxcon; j++) treef_coxcon << std::setprecision(save_tree_precision) << t_coxcon[j] << endl;

            msd_post(save_ctr) = mscale;
            bsd_post(save_ctr) = bscale1 - bscale0;
            b0_post(save_ctr) = bscale0;
            b1_post(save_ctr) = bscale1;
            redundt(save_ctr) = tmpbar/n;
            
            adrch_post(save_ctr, 0) = pi_con.a_drch;
            adrch_post(save_ctr, 1) = pi_coxcon.a_drch;
            adrch_post(save_ctr, 2) = pi_mod.a_drch;

            for (size_t k=0; k<p_m; ++k) {
                sigma_post(save_ctr,k) = sigma[k];
            }
            
            for (size_t k=0; k<(n*p_m); k++) {
                m_post(save_ctr,k) = allfit_con[k];
                mhat_post(save_ctr,k) = allfit_m[k];
            }
            
            for (size_t k=0; k<n; ++k) {
                coxcon_post(save_ctr,k) = allfit_coxcon[k];
                yhat_post(save_ctr,k) = allfit_y[k];
                coxtau_post(save_ctr,k) = allfit_mod[k*(p_m+1)+p_m];
                hzd_post(save_ctr,k) = cumhzd[k];
            }
            
            for (size_t k=0; k<n; k++) {
                for (size_t l=0; l<p_m; ++l) {
                    double bscale = (k<ntrt) ? bscale1 : bscale0;
                    b_post(save_ctr, k*(p_m)+l) = (bscale1 - bscale0)*allfit_mod[k*(p_m+1)+l]/bscale;
                }
            }
            
            for (size_t k=0; k<p_con; k++) {
                varcnt_con(save_ctr,k) = nv_con[k];
                scon_post(save_ctr, k) = PH_con.counts[k];//s_con[k];
            }
            
            for (size_t k=0; k<p_mod; k++) {
                varcnt_mod(save_ctr,k) = nv_mod[k];
                smod_post(save_ctr, k) = PH_mod.counts[k];//s_mod[k];
            }
            
            for (size_t k=0; k<p_coxcon; k++) {
                varcnt_coxcon(save_ctr,k) = nv_coxcon[k];
                scoxcon_post(save_ctr, k) = PH_coxcon.counts[k];//s_coxcon[k];
            }
            /*
            for (size_t k=0; k<G; ++k) {
                hzd_post(save_ctr,k) = h0[k];
            }*/
            
            for (size_t k=0; k<(ngrids+1); ++k) {
                hzd_survp(save_ctr,k) = cumhzd_survp[k];
            }
            
            for (size_t k=0; k<p_m*p_m; ++k) {
                covsigma_post(save_ctr, k) = arma::conv_to<std::vector<double>>::from(Covsigma.as_row())[k];
            }
            
            // do the prediction

            
            save_ctr += 1;
            //treef_con << std::setprecision(save_tree_precision) << save_ctr << endl;
            //treef_mod << std::setprecision(save_tree_precision) << save_ctr << endl;
            
        }
        
        logger.log("===================================");
        sprintf(logBuff, "MCMC iteration: %d of %d End", iIter+1, nd*thin+burn);
        logger.log(logBuff);
        sprintf(logBuff, "sigma %f, %f, mscale %f, bscale0 %f, bscale1 %f", sigma[0], sigma[1], mscale, bscale0, bscale1);
        logger.log(logBuff);
        logger.log("===================================");

        if (verbose_itr) {
            logger.getVectorHead(m, logBuff);
            Rcout << "      m : " << logBuff << "\n";
            
            logger.getVectorHead(allfit_m, logBuff);
            Rcout << " Current Fit_M: " << logBuff << "\n";
            
            logger.getVectorHead(truehzd_, logBuff);
            Rcout << "      hzd : " << logBuff << "\n";
            
            logger.getVectorHead(allfit_y, logBuff);
            Rcout << " Current Fit_hzd: " << logBuff << "\n";

            logger.getVectorHead(allfit_con, logBuff);
            Rcout << "allfit_con: " << logBuff << "\n";
            
            logger.getVectorHead(allfit_coxcon, logBuff);
            Rcout << "allfit_coxcon: " << logBuff << "\n";

            logger.getVectorHead(allfit_mod, logBuff);
            Rcout << "allfit_mod: " << logBuff << "\n";
            
            logger.getVectorHead(gamma, logBuff);
            Rcout << "gamma: " << logBuff << "\n";
            
            logger.getVectorHead(MHcounts_gamma, logBuff);
            Rcout << "MHcounts_gamma: " << logBuff << "\n";
            
            logger.getVectorHead(hzd, logBuff);
            Rcout << "baselinehzd: " << logBuff << "\n";
            
            logger.getVectorHead(cumhzd, logBuff);
            Rcout << "Cum_baselinehzd: " << logBuff << "\n";
        }
        

        
        
    } // end MCMC loop
        
    
    
    int time2 = time(&tp);
    Rcout << "\n=========================\n MCMC Complete \n=========================\n";
    Rcout << "time for loop: " << time2 - time1 << endl;
    
    
    t_mod.clear(); t_con.clear();t_coxcon.clear();
    delete [] allfit_m;
    delete [] allfit_y;
    delete[] allfit_mod;
    delete[] allfit_con;
    delete [] allfit_coxcon;
    delete[] r_mod;
    delete[] r_con;
    delete [] r_coxcon;
    delete[] ftemp_m;
    delete[] ftemp_y;
    delete[] ftemp_mny;
    delete [] weight_m;
    delete [] weight_y;
    delete [] weight_heter;
    delete [] pi_con.tau;
    delete [] pi_con.sigma;
    delete [] pi_mod.tau;
    delete [] pi_mod.sigma;
    delete [] hzd;
    delete [] cumhzd;
    delete [] gamma;
    delete [] MHcounts_gamma;
    delete [] cumhzd_survp;
    m.clear(); y.clear(); x_con.clear(); x_mod.clear(); x_coxcon.clear();

    treef_con.close();
    treef_mod.close();
    treef_coxcon.close();

    return (List::create(_["mhat_post"] = mhat_post, _["yhat_post"] = yhat_post, _["con_post"] = m_post, _["coxcon_post"] = coxcon_post, _["b_post"] = b_post, _["sigma_post"] = sigma_post, _["msd"] = msd_post, _["hzdsurvp_post"] = hzd_survp, _["b0"] = b0_post, _["b1"] = b1_post, _["varcnt_con_post"] = varcnt_con, _["varcnt_mod_post"] = varcnt_mod, _["varcnt_coxcon_post"] = varcnt_coxcon, _["hzd0_post"] = hzd_post, _["coxtau_post"] = coxtau_post, _["covsigma_post"] = covsigma_post, _["rddt_post"] = redundt, _["scon_post"] = scon_post, _["smod_post"] = smod_post, _["scoxcon_post"] = scoxcon_post));


}



