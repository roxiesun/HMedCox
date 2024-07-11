#include <iostream>
#include <RcppArmadillo.h>

#include "info.h"
#include "tree.h"
#include "swaprule.h"
#include "funcs.h"

using std::cout;
using std::endl;

bool swaprule(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv swapnv;
    swapnv.clear();
    x.getswap(swapnv);
    size_t Nswapnv = swapnv.size();
    logger.log("Attempting SwapRule");
    if (Nswapnv == 0) {
        logger.log("Rejecting SwapRule: no appropriate internal nodes for swapping");
        return false;
    }
    
    // randomly choose a swappable nodes
    size_t ni = floor(gen.uniform()*Nswapnv);
    tree::tree_p nx = swapnv[ni];
    
    // check whether the two kids of nx have the same rule
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    size_t lv = nx->getl()->getv();
    size_t rv = nx->getr()->getv();
    size_t lc = nx->getl()->getc();
    size_t rc = nx->getr()->getc();
    tree::npv bnv;
    std::vector<sinfo> oldsv;
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
    
    // if the two children share the same rule
    if ((lv==rv) && (lc==rc)) {
        // swap the rule
        tree::tree_p np = x.getptr(nx->nid());
        np->getl()->setv(oldv);
        np->getl()->setc(oldc);
        np->getr()->setv(oldv);
        np->getr()->setc(oldc);
        np->setv(lv);
        np->setc(lc);
        
        // check whether the swapped rule conflicts
        bool checkrule = CheckRule(np, xi, np->getv());
        if ((oldv!=lv) && checkrule) {
            checkrule = CheckRule(np, xi, oldv);
        }
        if (checkrule) { // if not conflict, do MH
            double logPrinew = logPriT(np, xi, pi);
            tree::npv bnvnew;
            std::vector<sinfo> newsv;
            allsuff(x, xi, di, phi, bnvnew, newsv);
            double logInew = 0.0;
            for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
             /*   Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
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
            double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
            
            if (gen.uniform() < alpha) {
                logger.log("Accepting SwapRule");
                return true;
            } else {
                np->setv(oldv);
                np->setc(oldc);
                np->getl()->setv(lv);
                np->getl()->setc(lc);
                np->getr()->setv(rv);
                np->getr()->setc(rc);
                logger.log("Rejecting SwapRule");
                return false;
            }
        } else {
            //if checkrule conflict, swap back
            np->setv(oldv);
            np->setc(oldc);
            np->getl()->setv(lv);
            np->getl()->setc(lc);
            np->getr()->setv(rv);
            np->getr()->setc(rc);
            logger.log("Rejecting SwapRule");
            return false;
        }
        
    } else { // if the two children have different rules, randomly choose one to swap the rule
        int lI, rI;
        lI=0;rI=0;
        if (!(nx->getl())) {
            lI=1;
        }
        if (!(nx->getr())) {
            rI=1;
        }
        if ((lI+rI)==0) { // double check
            //Rprintf("error in SwapRule: neither child has a rule to swap\n");
            return false;
        }
        tree::tree_p np = x.getptr(nx->nid());
        if ((lI+rI)==2) { // if both kid have rule
            double u = gen.uniform();
            if (u<0.5) {
                // swap with the left kid
                np->setv(lv);
                np->setc(lc);
                np->getl()->setv(oldv);
                np->getl()->setc(oldc);
            } else { // swap with the right kid
                np->setv(rv);
                np->setc(rc);
                np->getr()->setv(oldv);
                np->getr()->setc(oldc);
            }
        } else if (lI) {
            np->setv(lv);
            np->setc(lc);
            np->getl()->setv(oldv);
            np->getl()->setc(oldc);
        } else {
            np->setv(rv);
            np->setc(rc);
            np->getr()->setv(oldv);
            np->getr()->setc(oldc);
        }
        
        // checkrule
        bool checkrule = CheckRule(np, xi, np->getv());
        if ((oldv!=np->getv()) && checkrule) {
            checkrule = CheckRule(np, xi, oldv);
        }
        if (checkrule) { // not conflict, do MH
            double logPrinew = logPriT(np, xi, pi);
            tree::npv bnvnew;
            std::vector<sinfo> newsv;
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
            double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
            
            if (gen.uniform() < alpha) {
                logger.log("Accepting SwapRule");
                return true;
            } else {
                np->setv(oldv);
                np->setc(oldc);
                np->getl()->setv(lv);
                np->getl()->setc(lc);
                np->getr()->setv(rv);
                np->getr()->setc(rc);
                logger.log("Rejecting SwapRule");
                return false;
            }
        } else {
            //if checkrule conflict, swap back
            np->setv(oldv);
            np->setc(oldc);
            np->getl()->setv(lv);
            np->getl()->setc(lc);
            np->getr()->setv(rv);
            np->getr()->setc(rc);
            logger.log("Rejecting SwapRule");
            return false;
        }
    }
}

//swaprule with log-gamma par for the coxph model
bool swaprule_lg(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv swapnv;
    swapnv.clear();
    x.getswap(swapnv);
    size_t Nswapnv = swapnv.size();
    logger.log("Attempting SwapRule");
    if (Nswapnv == 0) {
        logger.log("Rejecting SwapRule: no appropriate internal nodes for swapping");
        return false;
    }
    
    // randomly choose a swappable nodes
    size_t ni = floor(gen.uniform()*Nswapnv);
    tree::tree_p nx = swapnv[ni];
    
    // check whether the two kids of nx have the same rule
    size_t oldv = nx->getv();
    size_t oldc = nx->getc();
    size_t lv = nx->getl()->getv();
    size_t rv = nx->getr()->getv();
    size_t lc = nx->getl()->getc();
    size_t rc = nx->getr()->getc();
    tree::npv bnv;
    std::vector<sinfo> oldsv;
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
        
        /*Rcpp::Rcout << "old:" << oldsv[i].n0 << " " << oldsv[i].ysize << endl;
        Rcpp::Rcout << oldsv[i].n << endl;
        Rcpp::Rcout << oldsv[i].sy << endl;*/
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
    
    // if the two children share the same rule
    if ((lv==rv) && (lc==rc)) {
        // swap the rule
        tree::tree_p np = x.getptr(nx->nid());
        np->getl()->setv(oldv);
        np->getl()->setc(oldc);
        np->getr()->setv(oldv);
        np->getr()->setc(oldc);
        np->setv(lv);
        np->setc(lc);
        
        // check whether the swapped rule conflicts
        bool checkrule = CheckRule(np, xi, np->getv());
        if ((oldv!=lv) && checkrule) {
            checkrule = CheckRule(np, xi, oldv);
        }
        if (checkrule) { // if not conflict, do MH
            double logPrinew = logPriT(np, xi, pi);
            tree::npv bnvnew;
            std::vector<sinfo> newsv;
            allsuff(x, xi, di, phi, bnvnew, newsv);
            
            tree::npv bnvnew_noexp;
            std::vector<sinfo> newsv_noexp;
            allsuff(x, xi, di_noexp, di.delta, bnvnew_noexp, newsv_noexp);
            
            double logInew = 0.0;
            for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
                
               /*  Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
                 Rcpp::Rcout << newsv[i].n << endl;
                 Rcpp::Rcout << newsv[i].sy << endl;*/
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
            double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
            
            if (gen.uniform() < alpha) {
                logger.log("Accepting SwapRule");
                delete [] r_noexp;
                return true;
            } else {
                np->setv(oldv);
                np->setc(oldc);
                np->getl()->setv(lv);
                np->getl()->setc(lc);
                np->getr()->setv(rv);
                np->getr()->setc(rc);
                logger.log("Rejecting SwapRule");
                delete [] r_noexp;
                return false;
            }
        } else {
            //if checkrule conflict, swap back
            np->setv(oldv);
            np->setc(oldc);
            np->getl()->setv(lv);
            np->getl()->setc(lc);
            np->getr()->setv(rv);
            np->getr()->setc(rc);
            logger.log("Rejecting SwapRule");
            delete [] r_noexp;
            return false;
        }
        
    } else { // if the two children have different rules, randomly choose one to swap the rule
        int lI, rI;
        lI=0;rI=0;
        if (!(nx->getl())) {
            lI=1;
        }
        if (!(nx->getr())) {
            rI=1;
        }
        if ((lI+rI)==0) { // double check
            //Rprintf("error in SwapRule: neither child has a rule to swap\n");
            return false;
        }
        tree::tree_p np = x.getptr(nx->nid());
        if ((lI+rI)==2) { // if both kid have rule
            double u = gen.uniform();
            if (u<0.5) {
                // swap with the left kid
                np->setv(lv);
                np->setc(lc);
                np->getl()->setv(oldv);
                np->getl()->setc(oldc);
            } else { // swap with the right kid
                np->setv(rv);
                np->setc(rc);
                np->getr()->setv(oldv);
                np->getr()->setc(oldc);
            }
        } else if (lI) {
            np->setv(lv);
            np->setc(lc);
            np->getl()->setv(oldv);
            np->getl()->setc(oldc);
        } else {
            np->setv(rv);
            np->setc(rc);
            np->getr()->setv(oldv);
            np->getr()->setc(oldc);
        }
        
        // checkrule
        bool checkrule = CheckRule(np, xi, np->getv());
        if ((oldv!=np->getv()) && checkrule) {
            checkrule = CheckRule(np, xi, oldv);
        }
        if (checkrule) { // not conflict, do MH
            double logPrinew = logPriT(np, xi, pi);
            tree::npv bnvnew;
            std::vector<sinfo> newsv;
            allsuff(x, xi, di, phi, bnvnew, newsv);
            
            tree::npv bnvnew_noexp;
            std::vector<sinfo> newsv_noexp;
            allsuff(x, xi, di_noexp, di.delta, bnvnew_noexp, newsv_noexp);
            
            double logInew = 0.0;
            for (tree::npv::size_type i=0; i!=bnvnew.size(); i++) {
                
               /*  Rcpp::Rcout << "newsv:" << newsv[i].n0 << " " << newsv[i].ysize << endl;
                 Rcpp::Rcout << newsv[i].n << endl;
                 Rcpp::Rcout << newsv[i].sy << endl;*/
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
            double alpha = std::min(1.0, exp(logPrinew + logInew - logPriold - logILold));
            
            if (gen.uniform() < alpha) {
                logger.log("Accepting SwapRule");
                delete [] r_noexp;
                return true;
            } else {
                np->setv(oldv);
                np->setc(oldc);
                np->getl()->setv(lv);
                np->getl()->setc(lc);
                np->getr()->setv(rv);
                np->getr()->setc(rc);
                logger.log("Rejecting SwapRule");
                delete [] r_noexp;
                return false;
            }
        } else {
            //if checkrule conflict, swap back
            np->setv(oldv);
            np->setc(oldc);
            np->getl()->setv(lv);
            np->getl()->setc(lc);
            np->getr()->setv(rv);
            np->getr()->setc(rc);
            logger.log("Rejecting SwapRule");
            delete [] r_noexp;
            return false;
        }
    }
}



