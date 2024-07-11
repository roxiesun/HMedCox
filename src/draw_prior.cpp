#include <iostream>
#include <RcppArmadillo.h>

#include "info.h"
#include "tree.h"
#include "bd.h"
#include "funcs.h"
#include "draw_prior.h"

using std::cout;
using std::endl;

// Modified from the MH-prior step of Linero and Du (2023), or SBART-MFM(https://github.com/theodds/SoftBART/blob/MFM/src/soft_bart.cpp#L964)

bool draw_prior_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers)
{
    tree::npv oldnv; // all nodes of x
    x.getnodes(oldnv);
    size_t oldsize = oldnv.size();
    logger.log("Attempting MH-prior step");
    if (oldsize == 0) {
        logger.log("Error in MH-prior: null tree");
        return false;
    }
    
    tree::tree_p nx = oldnv[0];
    tree::tree_p saveoldtree = &x;//x.getptr(nx->nid()); //copy the old tree
    //tree::tree_p np = x.getptr(nx->nid());
    
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
    SubtractTreeCounts(hypers, nx);
    
    tree newtree;
    //tree::tree_p newtree = new tree;
    newtree.p_mu = oldnv[0]->p_mu;
    newtree.mu = oldnv[0]->mu;
    GenBelow(&newtree, pi, di, xi, phi, hypers, gen, vs);
    
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(newtree, xi, di, phi, bnvnew, newsv);
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
    //Rcpp::Rcout << "LogILold: " << logILold << ", logInew:  " << logInew << endl;
    //logger.log("Attempting to Print the New Tree in MH-prior-step \n");
    //newtree.pr(xi);
    
    double alpha = std::min(1.0, exp(logInew -  logILold));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting MH-prior step");
        x.tonull();
        tree::tree_p np = x.getptr(1);
        tree::tree_p newp = newtree.getptr(1);
        np->v = newp->v;
        np->c = newp->c;
        np->mu = newp->mu;
        np->p_mu = newp->p_mu;
        
        if (newp->l) {
            tree::tree_p l = new tree;
            tree::tree_p r = new tree;
            x.cp(l, newp->l);
            x.cp(r, newp->r);
            np->l = l;
            np->r = r;
            l->p = np;
            r->p = np;
        }
        
        
        //x.cp(&x, &newtree);
        /*x.l = new tree;
        x.r = new tree;
        x.cp(x.l, newtree.l);
        (x.l)->p = &x;
        x.cp(x.r, newtree.r);
        (x.r)->p = &x;*/
        //x.cp(np, newtree);
        //ivcnt[oldv] -= 1;
        //ivcnt[v] += 1;
        //delete newtree;
        //delete saveoldtree;
        return true;
    } else {
        //x = *saveoldtree;
        //nx = saveoldtree;
        SubtractTreeCounts(hypers, &newtree);
        AddTreeCounts(hypers, saveoldtree);
        logger.log("Rejecting MH-prior step");
        //delete newtree;
        //delete saveoldtree;
        return  false;
    }
    
}

bool draw_prior_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers)
{
    tree::npv oldnv; // all nodes of x
    x.getnodes(oldnv);
    size_t oldsize = oldnv.size();
    logger.log("Attempting MH-prior step");
    if (oldsize == 0) {
        logger.log("Error in MH-prior: null tree");
        return false;
    }
    
    tree::tree_p nx = oldnv[0];
    tree::tree_p saveoldtree = &x; //copy the old tree
    //tree::tree_p np = x.getptr(nx->nid());
    
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
        if (di.p_y > 1) {
            /*
            Rcpp::Rcout << "old:" << oldsv[i].n0 << " " << oldsv[i].ysize << endl;
            Rcpp::Rcout << oldsv[i].n << endl;
            Rcpp::Rcout << oldsv[i].sy << endl;*/
            arma::vec oldsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                oldsy(k) = oldsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logILold += loglike_mvn(oldsv[i].n0, oldsy, Covsig, Covprior);
        }
        logILold += loglikelg(oldsv[i].sdelta, oldsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + oldsv_noexp[i].sy[0];
    }
    SubtractTreeCounts(hypers, nx);
    
    tree newtree;
    //tree::tree_p newtree = new tree;
    newtree.p_mu = oldnv[0]->p_mu;
    newtree.mu = oldnv[0]->mu;
    GenBelow(&newtree, pi, di, xi, phi, hypers, gen, vs);
    
    std::vector<sinfo> newsv;
    tree::npv bnvnew;
    allsuff(newtree, xi, di, phi, bnvnew, newsv);
    
    std::vector<sinfo> newsv_noexp;
    tree::npv bnvnew_noexp;
    allsuff(newtree, xi, di_noexp, di.delta, bnvnew_noexp, newsv_noexp);
    
    double logInew = 0.0;
    for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
        if (di.p_y > 1) {
            /*
            Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
            Rcpp::Rcout << newsv[i].n << endl;
            Rcpp::Rcout << newsv[i].sy << endl;*/
            arma::vec newsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                newsy(k) = newsv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            logInew += loglike_mvn(newsv[i].n0, newsy, Covsig, Covprior);
        }
        logInew += loglikelg(newsv[i].sdelta, newsv[i].sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + newsv_noexp[i].sy[0];
    }
    //Rcpp::Rcout << "LogILold: " << logILold << ", logInew:  " << logInew << endl;
    //logger.log("Attempting to Print the New Tree in MH-prior-step \n");
    //newtree.pr(xi);
    
    double alpha = std::min(1.0, exp(logInew -  logILold));
    
    if (gen.uniform() < alpha) {
        logger.log("Accepting MH-prior step");
        x.tonull();
        tree::tree_p np = x.getptr(1);
        tree::tree_p newp = newtree.getptr(1);
        np->v = newp->v;
        np->c = newp->c;
        np->mu = newp->mu;
        np->p_mu = newp->p_mu;
        
        if (newp->l) {
            tree::tree_p l = new tree;
            tree::tree_p r = new tree;
            x.cp(l, newp->l);
            x.cp(r, newp->r);
            np->l = l;
            np->r = r;
            l->p = np;
            r->p = np;
        }
        return true;
    } else {
        //x = *saveoldtree;
        //nx = saveoldtree;
        SubtractTreeCounts(hypers, &newtree);
        AddTreeCounts(hypers, saveoldtree);
        logger.log("Rejecting MH-prior step");
        newtree.tonull();//delete newtree;
        //delete saveoldtree;
        return  false;
    }
    
}
