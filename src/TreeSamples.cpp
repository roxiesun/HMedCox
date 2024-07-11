#include <RcppArmadillo.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "rand_gen.h"
#include "tree.h"
#include "info.h"
#include "funcs.h"
#include "bd.h"

using namespace Rcpp;

class TreeSamples{
public:
    bool init;
    size_t m,p,p_y,ndraws;
    xinfo xi;
    std::vector<std::vector<tree> > t;
    
    void load(CharacterVector treef_name_) {
        Rcout << "Loading...\n";
        std::string treef_name = as<std::string>(treef_name_);
        std::ifstream treef(treef_name.c_str());
        treef >> xi; // load cutpoints read it in?
        treef >> m; // number of trees
        Rcout << "ntrees: " << m << endl;
        treef >> p; //dim of covariates
        Rcout << "p: " << p << endl;
        treef >> p_y; //dim of response
        Rcout << "p_y: " << p_y << endl;
        treef >> ndraws; // number of draws from the posterior
        Rcout << "ndraws: " << ndraws <<endl;
        
        t.resize(ndraws,std::vector<tree>(m));
        Rcout << "Resize completed. Now reading in the trees." <<endl;
        for (size_t i=0; i<ndraws; i++) {
            for (size_t j=0; j<m; j++) {
                treef >> t[i][j]; // pointer to the jth tree in ith draw
            }
        }
        Rcout << "done" << endl;
        init = true;
        
        treef.close();
    }
    
    NumericMatrix predict(NumericMatrix x_) {
        size_t n = x_.ncol(); //sample size?
        NumericMatrix ypred(ndraws,n*p_y);
        if (init) {
            std::vector<double> x;
            for (NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) {
                x.push_back(*it);
            }
            
            dinfo di;
            di.N=n; di.p=p; di.x=&x[0]; di.y=0; di.p_y = p_y; di.delta=0;
            
            for (size_t k=0; k<n; ++k) {
                for (size_t i=0; i<ndraws; i++) {
                    for (size_t l=0; l<p_y; ++l) {
                        ypred(i,k*p_y+l) += fit_i(k, t[i], xi, di)[l];
                    }
                }
            }
        } else {
            Rcout << "Uninitialized" << "\n";
        }
        return ypred; //posterior predictive?
    }
    
    // predictions for multiplicative trees? (precision?)
    NumericMatrix predict_prec(NumericMatrix x_){
        size_t n = x_.ncol();
        NumericMatrix ypred(ndraws,n*p_y);
        ypred.fill(1.0);
        if (init) {
            std::vector<double> x;
            for (NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) {
                x.push_back(*it);
            }
            
            dinfo di;
            di.N=n; di.p=p; di.x=&x[0]; di.y=0; di.p_y=p_y; di.delta=0;
            
            for (size_t k=0; k<n; ++k) {
                for (size_t i=0; i<ndraws; i++) {
                    for (size_t l=0; l<p_y; ++l) {
                        ypred(i,k*p_y+l) *= fit_i_mult(k, t[i], xi, di)[l];
                    }
                }
            }
        } else {
            Rcout << "Uninitialized" << "\n";
        }
        return ypred;
    }
    
    // prediction from the ith mcmc iterate (or the ith draw
    NumericMatrix predict_i(NumericMatrix x_, size_t i) {
        size_t n = x_.ncol();
        NumericMatrix ypred(1,n*p_y);
        if (init) {
            std::vector<double> x;
            for (NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) {
                x.push_back(*it);
            }
            
            dinfo di;
            di.N=n; di.p=p; di.x=&x[0]; di.y=0; di.p_y=p_y; di.delta=0;
            
            for (size_t k=0; k<n; ++k) {
                for (size_t l=0; l<p_y; ++l) {
                    ypred(0,k*p_y+l) += fit_i(k, t[i], xi, di)[l];
                }
            }
        } else {
            Rcout << "Uninitialzed" << "\n";
        }
        return ypred;
    }
    
    // prediction precision from the ith mcmc iterate
    NumericMatrix predict_prec_i(NumericMatrix x_, size_t i) {
        size_t n = x_.ncol();
        NumericMatrix ypred(1,n*p_y); ypred.fill(1.0);
        if (init) {
            std::vector<double> x;
            for (NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) x.push_back(*it);
            
            dinfo di;
            di.N=n; di.p=p; di.x=&x[0]; di.y=0; di.p_y=p_y; di.delta=0;
            
            for (size_t k=0; k<n; ++k) {
                for (size_t l=0; l<p_y; ++l) {
                    ypred(0,k*p_y+l) *= fit_i_mult(k, t[i], xi, di)[l];
                }
            }
        } else {
            Rcout << "Uninitialized" <<"\n";
        }
        return ypred;
    }
    
    //constructor
    TreeSamples() {};
    
};

RCPP_MODULE(TreeSamples) {
    class_<TreeSamples>("TreeSamples")
    .constructor()
    .method("load", &TreeSamples::load)
    .method("predict", &TreeSamples::predict)
    .method("predict_prec", &TreeSamples::predict_prec)
    .method("predict_i", &TreeSamples::predict_i)
    .method("predict_prec_i", &TreeSamples::predict_prec_i)
    ;
}
