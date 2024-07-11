#ifndef GUARD_info_h
#define GUARD_info_h

#include<vector>


// data info, define constants
class dinfo {
public:
    dinfo() {p = 0; p_y = 0; N = 0; x = 0; y = 0; delta = 0;} // constructor?
    size_t p;   // dim of covariates
    size_t p_y; // dim of response
    size_t N;   // number of observations
    double *x;  /* so that j_th var of the i_th subject is *(x + p*i + j), saved as a vector? */
    double *y; // y_i is *(y+i) or y[i]
    double *delta;
};

// prior and mcmc information
class pinfo {
public:
    pinfo() {pbd = 0.5; pb = 0.25; pchange = 0.4; pswap = 0.1; alpha = 0.95; beta = 0.5; tau = 0; lg_alpha = 0; lg_beta = 0; sigma = 0; minperbot = 5;
        taccept = 0; tavgd = 0; tmaxd = 0;
        a_drch = 1.; }//s=0;}
    //mcmc
    double pbd;  // prob of birth/death
    double pb;   // prob of birth
    // But what about swap and change?
    double pchange;
    double pswap;
    
    // prior info
    double alpha; // base hyper
    double beta;  // power hyper?
    double* tau;   // true homogenous treatment effects? or what hyper? node variance hyperparameter?
    double lg_alpha; double lg_beta;
    // sigma
    double* sigma; //error variance
    // for variable selection with dirichlet prior
    double a_drch;
    //double* s; // par vec for the dirichlet prior
    
    //tree min obs
    size_t minperbot; // what for?
    
    // mcmc settings
    std::vector< std::vector<double> > cvpm;  // change of variable probability matrix
    unsigned int taccept;  // acceptance count of tree proposals
    unsigned int tproposal; // number of tree proposals, to check the acceptance rate?
    unsigned int tavgd; // average tree depth
    unsigned int tmaxd; // maximum tree depth
};

// sufficient statistics for 1 node
class sinfo {
public:
    sinfo(): n(1), sy(1) {n0 = 0.0; ysize = 1; sdelta = 0.0;} // constructor, originally n=0
    double n0;  // unweighted sample counts.
    std::vector<double> n;//double n;
    size_t ysize;
    std::vector<double> sy;  // sum y of this node?
    double sdelta;
};


#endif /* info_h */
