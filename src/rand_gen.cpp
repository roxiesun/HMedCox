#include <cmath>
#include "rand_gen.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::plugins("cpp11")]]

// standard normal truncated > lo
double rtnormlo0(double lo){
    double x;
    if (lo < 0) {
        x = R::rnorm(0.0, 1.0);
        while (x < lo) x = R::rnorm(0.0, 1.0);
    } else {
        double a = 0.5*(lo + sqrt(lo*lo + 4.0));
        x = R::rexp(1.0/a) + lo;   // accept-reject sampling?
        double u = R::runif(0.0, 1.0);
        double diff = (x - a);
        double r = exp(-0.5*diff*diff);
        while (u > r) {
            x = R::rexp(1.0/a) + lo;  // the proposal; u > r is because g = (1/a)exp[(-1/a)*x)] < 1
            u = R::runif(0.0, 1.0);
            diff = (x - a);
            r = exp(-0.5*diff*diff);
        }
    }
    return x;
}

double rtnormlo1(double mean, double lo){
    return mean + rtnormlo0(lo - mean);
}

double rtnormlo(double mean, double sd, double lo){
    double lotr = (lo - mean)/sd;
    return mean + rtnormlo0(lotr)*sd;
}

double rtnormhi1(double mean, double lo){
    return -rtnormlo1(-mean, -lo);
}

// sample from mvtnorm posterior

arma::vec rmvnorm_post(arma::vec &m, arma::mat &Phi){
    arma::mat R = arma::chol(Phi);
    arma::vec mu = arma::solve(arma::trimatu(R), arma::solve(arma::trimatl(R.t()),m)); //LU decomposition?
    arma::vec z(mu.size());
    for (size_t p = 0; p < mu.size(); ++p) {
        z(p) = R::rnorm(0.0, 1.0);
    }
    arma::vec res = mu + arma::solve(arma::trimatu(arma::chol(Phi)),z);
    // 2nd term residual?
    return res;
}

/*
double rlgam(double shape) {
  if(shape >= 0.1) return log(R::rgamma(shape,1));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = R::runif(0., 1.);
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(R::runif(0., 1.)) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(R::runif(0., 1.))) break;
  } while(true);

  // Rcout << "Sample: " << -z/a << "\n";

  return -z/a;
} */
