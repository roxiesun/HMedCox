#ifndef _RAND_GEN_H
#define _RAND_GEN_H
#include <RcppArmadillo.h>



using std::vector;

class RNG
{
public:
    // continuous distributions, generate 1
    double uniform(double x = 0.0, double y = 1.0)   // sample and return
    {return R::runif(x,y);}
    
    double normal(double mu = 0.0, double sd = 1.0)
    {return R::rnorm(mu,sd);}
    
    double exp(double rate = 1.0)
    {return R::rexp(1/rate);}
    
    double gamma(double shape = 1, double scale = 1)
    {return R::rgamma(shape,1)*scale;}
    
    //double gamma(double shape = 1, double scale = 1)
    //{return R::rgamma(shape,scale);}//*scale;}
    
    double loggamma(double shape = 1, double scale = 1){
        return log(R::rgamma(shape,1)*scale);
    }
    
    double chisq(double df)
    {return R::rchisq(df);
        // return gamma(df/2.0, 0.5);
    }
    
    double halfcauchy(double sigma)
    {
        return std::abs(R::rcauchy(0.0, sigma));
    }
    
    double beta(double a1, double a2)
    {const double x1 = gamma(a1,1); return (x1/(x1 + gamma(a2,1)));}
    
    int binom(int size, double prob)
    {return R::rbinom(size, prob);}
    
    arma::mat wishart(arma::mat S, int df)
    {
        return arma::wishrnd(S, df);
    }
    
    arma::mat mvnorm(arma::vec m, arma::mat sigma)
    {
        int ncols = sigma.n_cols;
        arma::mat Y = arma::randn(1, ncols);
        return arma::repmat(m, 1, 1).t() + Y * arma::chol(sigma);
    }
    
    void uniform(vector<double>& res, double x = 0.0, double y = 1.0){    // assign value
        for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
            *i = uniform(x,y);
    }
    
    void normal(vector<double>& res, double mu = 0.0, double sd = 1.0){
        for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
            *i = normal(mu, sd);
    }
    
    void gamma(vector<double>& res, double shape = 1, double scale = 1){
        for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
            *i = gamma(shape, scale);
    }
    
    void chisq(vector<double>& res, double df){
        for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
            *i = chisq(df);
    }
    
    void beta(vector<double>& res, double a1, double a2){
        for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
            *i = beta(a1,a2);
    }
    
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
    }
};   // class for random number generator,but why not Rcpp::rnorm(n, mu, sd)?


inline double znorm(){          // will be freq used
    return R::rnorm(0.0, 1.0);
}

template<class T>
int rdisc_log(T &logweights, int n = -1){
    if (n == -1) n = logweights.size();
    typename T::iterator itb = logweights.begin();
    typename T::iterator ite = logweights.begin() + n;
    
    double m = *std::max_element(itb, ite);
    double s = 0;
    std::vector<double> weights(n, 0.0);  // weights = rep(0,n)?
    for (int i =0; i < n; i++) {
        weights[i] = exp(logweights[i] - m);
        s += weights[i];                // so s is the sum of weights?
    }
    
    double u = s*R::runif(0.0, 1.0);
    double cs = weights[0];
    int i = 0;
    while ((u > cs)&(i < n)) {
        ++i;
        cs += weights[i];
    }
    return i;
}

template<class T>
int rdisc_log_inplace(T &logweights, int n = -1, double u = -1.0){
    if (n == -1) n = logweights.size();
    if (u == -1) u = R::runif(0.0, 1.0);
    typename T::iterator itb = logweights.begin();
    typename T::iterator ite = logweights.begin() + n;
    
    double m = *std::max_element(itb, ite);
    double s = 0;
    for (int i =0; i < n; ++i) {
        logweights[i] = exp(logweights[i] - m);
        s += logweights[i];                // so s is the sum of log_weights?
    }
    u = s*u;
    double cs = logweights[0];
    int i = 0;
    while ((u > cs)&(i < n)) {
        ++i;
        cs += logweights[i];
    }
    return i;
}

// what are these for? various normal distribution trucated at low/high thresholds
double rtnormlo0(double lo);
double rtnormlo1(double mean, double lo);
double rtnormhi1(double mean, double lo);
double rtnormlo(double mean, double sd, double lo);
// from SBART (Linearo and Yang, 2018), to faciliate sampling from Dirichlet
//double rlgam(double shape);

//sample from N(Phi^(-1)m, Phi^(-1))
arma::vec rmvnorm_post(arma::vec &m, arma::mat &Phi);

//sample from log-gamma distribution
/*double rlgamma(double shape = 1, double scale = 1){
    return std::log(R::rgamma(shape,1)*scale);
}

double rhcauchy(double sigma){
    return std::abs(R::rcauchy(0.0, sigma));
}*/

#endif /* rand_gen_h */
