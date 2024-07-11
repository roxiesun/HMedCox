#include <iostream>
#include <RcppArmadillo.h>

#include "info.h"
#include "tree.h"
#include "changerule.h"
#include "funcs.h"

using std::cout;
using std::endl;

bool changerule(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv internv; // not bot nodes
    x.getnbots(internv);
    size_t Ninternv = internv.size();
    logger.log("Attempting ChangeRule");
    if (Ninternv == 0) {
        logger.log("Rejecting ChangeRule: no appropriate internal nodes for changing");
        return false;
    }
    
    size_t ni = floor(gen.uniform()*Ninternv);//randomly choose a not bot nodes to change its rule;
    tree::tree_p nx = internv[ni];
    
    // find a good var for the chosen INTERNAL node nx
    std::vector<size_t> goodintvars;
    getinternalvars(nx, xi, goodintvars);
    size_t vi = floor(gen.uniform()*goodintvars.size());
    size_t v = goodintvars[vi];
    
    // draw cutpoint uniformly for node nx with var v
    int L,U;
    L=0; U=xi[v].size()-1;
    //nx->region(v, &L, &U); not true, because our internal node may have left/right children that also uses v,instead, we use
    getpertLU(nx, v, xi, &L, &U);
    size_t c = L + floor(gen.uniform()*(U-L+1));
    
    // prepare for MH
    // note that the proposal distribution is identical for the changerule move??, so we only need to consider the logIlike,
    std::vector<sinfo> oldsv;
    tree::npv bnv;
    allsuff(x, xi, di, phi, bnv, oldsv);
    double logILold = 0.0;
    for (tree::npv::size_type i=0 ; i!=bnv.size(); i++) {
        /*
        Rcpp::Rcout << "old:" << oldsv[i].n0 << " " << oldsv[i].ysize << endl;
        Rcpp::Rcout << oldsv[i].n << endl;
        Rcpp::Rcout << oldsv[i].sy << endl;*/
        arma::vec oldsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            oldsy(k) = oldsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        logILold += loglike_mvn(oldsv[i].n0, oldsy, Covsig, Covprior);
        /*
        for (size_t k=0; k<di.p_y; ++k) {
            logILold += loglike(oldsv[i].n[k], oldsv[i].sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        //logILold += loglike(oldsv[i].n, oldsv[i].sy, pi.sigma, pi.tau);
    }
    double logPriold = logPriT(nx, xi, pi);
    
    // save current var and cutoff
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    
    // change to new rule
    tree::tree_p np = x.getptr(nx->nid());
    np->setv(v);
    np->setc(c);
    // because I doubt nx is just a copy of orginal tree node and thus setting v&c for nx might not work.
    //nx->setv(v);
    //nx->setc(c);
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(x, xi, di, phi, bnvnew, newsv);
    double logInew = 0.0;
    for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
        /*
        Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
        Rcpp::Rcout << newsv[i].n << endl;
        Rcpp::Rcout << newsv[i].sy << endl;*/
        arma::vec newsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            newsy(k) = newsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        logInew += loglike_mvn(newsv[i].n0, newsy, Covsig, Covprior);
        /*
        for (size_t k=0; k<di.p_y; ++k) {
            logInew += loglike(newsv[i].n[k], newsv[i].sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        //logInew += loglike(newsv[i].n, newsv[i].sy, pi.sigma, pi.tau);
    }
    double logPrinew = logPriT(np, xi, pi);
    
    double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting ChangeRule");
        ivcnt[oldv] -= 1;
        ivcnt[v] += 1;
        return true;
    } else {
        np->setv(oldv);
        np->setc(oldc);
        logger.log("Rejecting ChangeRule");
        return  false;
    }
}

bool changerule_lg(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv internv; // not bot nodes
    x.getnbots(internv);
    size_t Ninternv = internv.size();
    logger.log("Attempting ChangeRule");
    if (Ninternv == 0) {
        logger.log("Rejecting ChangeRule: no appropriate internal nodes for changing");
        return false;
    }
    
    size_t ni = floor(gen.uniform()*Ninternv);//randomly choose a not bot nodes to change its rule;
    tree::tree_p nx = internv[ni];
    
    // find a good var for the chosen INTERNAL node nx
    std::vector<size_t> goodintvars;
    getinternalvars(nx, xi, goodintvars);
    size_t vi = floor(gen.uniform()*goodintvars.size());
    size_t v = goodintvars[vi];
    
    // draw cutpoint uniformly for node nx with var v
    int L,U;
    L=0; U=xi[v].size()-1;
    //nx->region(v, &L, &U); not true, because our internal node may have left/right children that also uses v,instead, we use
    getpertLU(nx, v, xi, &L, &U);
    size_t c = L + floor(gen.uniform()*(U-L+1));
    
    // prepare for MH
    // note that the proposal distribution is identical for the changerule move??, so we only need to consider the logIlike,
    std::vector<sinfo> oldsv;
    tree::npv bnv;
    allsuff(x, xi, di, phi, bnv, oldsv);
    
    std::vector<sinfo> oldsv_noexp;
    tree::npv bnv_noexp;
    dinfo di_noexp = di;
    double* r_noexp = new double[di.N];
    for (size_t k=0; k<di.N; ++k) {
        r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
    }
    di_noexp.y = r_noexp;
    di_noexp.p_y = 1;
    allsuff(x, xi, di_noexp, di.delta, bnv_noexp, oldsv_noexp);
    
    double logILold = 0.0;
    for (tree::npv::size_type i=0 ; i!=bnv.size(); i++) {
        if (di.p_y>1) {
            arma::vec oldsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                oldsy(k) = oldsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logILold += loglike_mvn(oldsv[i].n0, oldsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                logILold += loglike(oldsv[i].n[k], oldsv[i].sy[k], pi.sigma[k], pi.tau[k]);
            }*/
        }
        logILold += loglikelg(oldsv[i].sdelta, oldsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + oldsv_noexp[i].sy[0];
    }
    double logPriold = logPriT(nx, xi, pi);
    
    // save current var and cutoff
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    
    // change to new rule
    tree::tree_p np = x.getptr(nx->nid());
    np->setv(v);
    np->setc(c);
    // because I doubt nx is just a copy of orginal tree node and thus setting v&c for nx might not work.
    //nx->setv(v);
    //nx->setc(c);
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(x, xi, di, phi, bnvnew, newsv);
    
    std::vector<sinfo> newsv_noexp;
    tree::npv bnvnew_noexp;
    allsuff(x, xi, di_noexp, di.delta, bnvnew_noexp, newsv_noexp);
    
    double logInew = 0.0;
    for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
        if (di.p_y>1) {
            arma::vec newsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                newsy(k) = newsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logInew += loglike_mvn(newsv[i].n0, newsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                logInew += loglike(newsv[i].n[k], newsv[i].sy[k], pi.sigma[k], pi.tau[k]);
            }*/
        }
        logInew += loglikelg(newsv[i].sdelta, newsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + newsv_noexp[i].sy[0];
    }
    double logPrinew = logPriT(np, xi, pi);
    
    double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting ChangeRule");
        ivcnt[oldv] -= 1;
        ivcnt[v] += 1;
        delete [] r_noexp;
        return true;
    } else {
        np->setv(oldv);
        np->setc(oldc);
        logger.log("Rejecting ChangeRule");
        delete [] r_noexp;
        return  false;
    }
    
}


// with variable selection included as an option
bool changerule_withvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers)
{
    tree::npv internv; // not bot nodes
    x.getnbots(internv);
    size_t Ninternv = internv.size();
    logger.log("Attempting ChangeRule");
    if (Ninternv == 0) {
        logger.log("Rejecting ChangeRule: no appropriate internal nodes for changing");
        return false;
    }
    
    size_t ni = floor(gen.uniform()*Ninternv);//randomly choose a not bot nodes to change its rule;
    tree::tree_p nx = internv[ni];
    
    // find a good var for the chosen INTERNAL node nx
    std::vector<size_t> goodintvars;
    getinternalvars(nx, xi, goodintvars);
    size_t vi = floor(gen.uniform()*goodintvars.size());
    size_t v = goodintvars[vi];
    
    if (vs) {
        /*
        vi = sample_class(probs, gen);//floor(gen.uniform()*goodvars.size());
        while (std::find(goodintvars.begin(), goodintvars.end(), vi)==goodintvars.end()) {
                vi = sample_class(probs, gen);
        }
        v = vi;*/
        vi = hypers.SampleVar(gen);
        while (std::find(goodintvars.begin(), goodintvars.end(), vi)==goodintvars.end()) {
                vi = hypers.SampleVar(gen);
        }
        v = vi;
    }
    
    // draw cutpoint uniformly for node nx with var v
    int L,U;
    L=0; U=xi[v].size()-1;
    //nx->region(v, &L, &U); not true, because our internal node may have left/right children that also uses v,instead, we use
    getpertLU(nx, v, xi, &L, &U);
    size_t c = L + floor(gen.uniform()*(U-L+1));
    
    // prepare for MH
    // note that the proposal distribution is identical for the changerule move??, so we only need to consider the logIlike,
    std::vector<sinfo> oldsv;
    tree::npv bnv;
    allsuff(x, xi, di, phi, bnv, oldsv);
    double logILold = 0.0;
    for (tree::npv::size_type i=0 ; i!=bnv.size(); i++) {
        /*
        Rcpp::Rcout << "old:" << oldsv[i].n0 << " " << oldsv[i].ysize << endl;
        Rcpp::Rcout << oldsv[i].n << endl;
        Rcpp::Rcout << oldsv[i].sy << endl;*/
        arma::vec oldsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            oldsy(k) = oldsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        logILold += loglike_mvn(oldsv[i].n0, oldsy, Covsig, Covprior);
        /*
        for (size_t k=0; k<di.p_y; ++k) {
            logILold += loglike(oldsv[i].n[k], oldsv[i].sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        //logILold += loglike(oldsv[i].n, oldsv[i].sy, pi.sigma, pi.tau);
    }
    double logPriold = logPriT(nx, xi, pi);
    
    // save current var and cutoff
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    
    double cutp_likelihood = cutpoint_likelihood(nx, xi);
    int oldL, oldU;
    oldL = 0; oldU = xi[oldv].size() - 1;
    getpertLU(nx, oldv, xi, &oldL, &oldU);
    double bw_trans = 1.0/(xi[oldv][oldU] - xi[oldv][oldL]);
    
    // change to new rule
    tree::tree_p np = x.getptr(nx->nid());
    np->setv(v);
    np->setc(c);
    // because I doubt nx is just a copy of orginal tree node and thus setting v&c for nx might not work.
    //nx->setv(v);
    //nx->setc(c);
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(x, xi, di, phi, bnvnew, newsv);
    double logInew = 0.0;
    for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
        /*
        Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
        Rcpp::Rcout << newsv[i].n << endl;
        Rcpp::Rcout << newsv[i].sy << endl;*/
        arma::vec newsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            newsy(k) = newsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        logInew += loglike_mvn(newsv[i].n0, newsy, Covsig, Covprior);
        /*
        for (size_t k=0; k<di.p_y; ++k) {
            logInew += loglike(newsv[i].n[k], newsv[i].sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        //logInew += loglike(newsv[i].n, newsv[i].sy, pi.sigma, pi.tau);
    }
    double logPrinew = logPriT(np, xi, pi);
    
    double cutp_likelihood_new = cutpoint_likelihood(np, xi);
    double fw_trans = 1.0 / (xi[v][U] - xi[v][L]);
    
    
    double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold + log(cutp_likelihood_new) + log(bw_trans) - log(cutp_likelihood) - log(fw_trans)));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting ChangeRule");
        ivcnt[oldv] -= 1;
        ivcnt[v] += 1;
        hypers.SwitchVar(oldv, v);
        return true;
    } else {
        np->setv(oldv);
        np->setc(oldc);
        logger.log("Rejecting ChangeRule");
        return  false;
    }
}


bool changerule_lgwithvs(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers)
{
    tree::npv internv; // not bot nodes
    x.getnbots(internv);
    size_t Ninternv = internv.size();
    logger.log("Attempting ChangeRule");
    if (Ninternv == 0) {
        logger.log("Rejecting ChangeRule: no appropriate internal nodes for changing");
        return false;
    }
    
    size_t ni = floor(gen.uniform()*Ninternv);//randomly choose a not bot nodes to change its rule;
    tree::tree_p nx = internv[ni];
    
    // find a good var for the chosen INTERNAL node nx
    std::vector<size_t> goodintvars;
    getinternalvars(nx, xi, goodintvars);
    size_t vi = floor(gen.uniform()*goodintvars.size());
    size_t v = goodintvars[vi];
    
    if (vs) {
        /*
        vi = sample_class(probs, gen);//floor(gen.uniform()*goodvars.size());
        while (std::find(goodintvars.begin(), goodintvars.end(), vi)==goodintvars.end()) {
                vi = sample_class(probs, gen);
        }
        v = vi;*/
        vi = hypers.SampleVar(gen);
        while (std::find(goodintvars.begin(), goodintvars.end(), vi)==goodintvars.end()) {
                vi = hypers.SampleVar(gen);
        }
        v = vi;
    }
    
    // draw cutpoint uniformly for node nx with var v
    int L,U;
    L=0; U=xi[v].size()-1;
    //nx->region(v, &L, &U); not true, because our internal node may have left/right children that also uses v,instead, we use
    getpertLU(nx, v, xi, &L, &U);
    size_t c = L + floor(gen.uniform()*(U-L+1));
    
    // prepare for MH
    // note that the proposal distribution is identical for the changerule move??, so we only need to consider the logIlike,
    std::vector<sinfo> oldsv;
    tree::npv bnv;
    allsuff(x, xi, di, phi, bnv, oldsv);
    
    std::vector<sinfo> oldsv_noexp;
    tree::npv bnv_noexp;
    dinfo di_noexp = di;
    double* r_noexp = new double[di.N];
    for (size_t k=0; k<di.N; ++k) {
        r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
    }
    di_noexp.y = r_noexp;
    di_noexp.p_y = 1;
    allsuff(x, xi, di_noexp, di.delta, bnv_noexp, oldsv_noexp);
    
    double logILold = 0.0;
    for (tree::npv::size_type i=0 ; i!=bnv.size(); i++) {
        if (di.p_y>1) {
            arma::vec oldsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                oldsy(k) = oldsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logILold += loglike_mvn(oldsv[i].n0, oldsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                logILold += loglike(oldsv[i].n[k], oldsv[i].sy[k], pi.sigma[k], pi.tau[k]);
            }*/
        }
        logILold += loglikelg(oldsv[i].sdelta, oldsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + oldsv_noexp[i].sy[0];
    }
    double logPriold = logPriT(nx, xi, pi);
    
    // save current var and cutoff
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    
    double cutp_likelihood = cutpoint_likelihood(nx, xi);
    int oldL, oldU;
    oldL = 0; oldU = xi[oldv].size() - 1;
    getpertLU(nx, oldv, xi, &oldL, &oldU);
    double bw_trans = 1.0/(xi[oldv][oldU] - xi[oldv][oldL]);
    
    // change to new rule
    tree::tree_p np = x.getptr(nx->nid());
    np->setv(v);
    np->setc(c);
    // because I doubt nx is just a copy of orginal tree node and thus setting v&c for nx might not work.
    //nx->setv(v);
    //nx->setc(c);
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(x, xi, di, phi, bnvnew, newsv);
    
    std::vector<sinfo> newsv_noexp;
    tree::npv bnvnew_noexp;
    allsuff(x, xi, di_noexp, di.delta, bnvnew_noexp, newsv_noexp);
    
    double logInew = 0.0;
    for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
        if (di.p_y>1) {
            arma::vec newsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                newsy(k) = newsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logInew += loglike_mvn(newsv[i].n0, newsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                logInew += loglike(newsv[i].n[k], newsv[i].sy[k], pi.sigma[k], pi.tau[k]);
            }*/
        }
        logInew += loglikelg(newsv[i].sdelta, newsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + newsv_noexp[i].sy[0];
    }
    double logPrinew = logPriT(np, xi, pi);
    
    double cutp_likelihood_new = cutpoint_likelihood(np, xi);
    double fw_trans = 1.0 / (xi[v][U] - xi[v][L]);
    
    double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold + log(cutp_likelihood_new) + log(bw_trans) - log(cutp_likelihood) - log(fw_trans)));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting ChangeRule");
        ivcnt[oldv] -= 1;
        ivcnt[v] += 1;
        hypers.SwitchVar(oldv, v);
        delete [] r_noexp;
        return true;
    } else {
        np->setv(oldv);
        np->setc(oldc);
        logger.log("Rejecting ChangeRule");
        delete [] r_noexp;
        return  false;
    }
    
}
