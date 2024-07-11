#include <iostream>
#include <RcppArmadillo.h>

#include "info.h"
#include "tree.h"
#include "bd.h"
#include "funcs.h"

using std::cout;
using std::endl;

/*Note: going from state x to state y
 if return true: birth/death accept*/

bool bd(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv goodbots; // bot nodes that can split
    double PBx = getpb(x, xi, pi, goodbots);
    
    if (gen.uniform() < PBx) {
        logger.log("Attempting Birth");
        
        // draw proposal
        
        // uniformly draw bottom node: choose node index from goodbots
        size_t ni = floor(gen.uniform()*goodbots.size()); // rounddown
        tree::tree_p nx = goodbots[ni];
        
        // draw variable v, uniformly
        std::vector<size_t> goodvars;
        getgoodvars(nx, xi, goodvars); // get variable this node can split on
        size_t vi = floor(gen.uniform()*goodvars.size());
        size_t v = goodvars[vi];
      /*  cout << goodvars.size() << "vars at choice" << endl;
        cout << vi  << "th goodvar is chosen" << endl;
        cout << v << "th predictor is chosen " << endl;*/
        
        // draw cutpoint, uniformly
        int L,U;
        L=0; U=xi[v].size()-1;
        nx->region(v, &L, &U);
        size_t c = L + floor(gen.uniform()*(U-L+1));
        // U-L+1 is the number of available split points
        
        //-------------------
        // prepare for Metropolis hastings
        double Pbotx = 1.0/goodbots.size(); // proposal dist/probability of choosing nx;
        size_t dnx = nx->depth();
        double PGnx = pi.alpha/pow(1.0+dnx, pi.beta); // prior prob of growing at nx;
        
        double PGly,PGry; // prior probs of growing at new children
        if (goodvars.size()>1) {
            PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            PGry = PGly;
        } else { // have only one v to work with
            if ((int)(c-1)<L) { // v exhausted in new left child l, new upper limit would be c-1
                PGly = 0.0;
            } else {
                PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
            if (U<(int)(c+1)) { // v exhausted in new right child r, new lower limit would be c+1
                PGry = 0.0;
            } else {
                PGry = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
        }
        
        double PDy; // prob of proposing death at y;
        if (goodbots.size()>1) { // can birth at y because splittable nodes left
            PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
        } else { //nx is the only splittable node
            if ((PGry==0) && (PGly==0)) { //cannot birth at y
                PDy = 1.0;
            } else { // y can birth can either l or r
                PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
            }
        }
        
        double Pnogy; // death prob of choosing the nog node at y
        size_t nnogs = x.nnogs();
        tree::tree_cp nxp = nx->getp();
        if (nxp==0) {
            Pnogy = 1.0;
        } else {
            if (nxp->isnog()) { // is parent is a nog, nnogs same at state x and y
                Pnogy = 1.0/nnogs;
            } else {
                Pnogy = 1.0/(nnogs+1.0);
            }
        } // we need Pnogy because we have to choose a nog to perform death at state y.
        
        //-----------
        //compute suff stats
        sinfo sl,sr;
        //sl.ysize = di.p_y; sr.ysize = di.p_y;
        //sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        //sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
        getsuffBirth(x, nx, v, c, di.p_y, xi, di, phi, sl, sr);
        
        /*
        Rcpp::Rcout << "left" << sl.n0  << " " << sl.ysize << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sy << endl;
        
        Rcpp::Rcout << "right" << sr.n0  <<  sr.ysize << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sy << endl;*/
        //---------
        //compute alpha
        double alpha=0.0, alpha1=0.0, alpha2=0.0;
        double lill=0.0, lilr=0.0, lilt=0.0;
        arma::vec slsy(di.p_y, arma::fill::zeros);
        arma::vec srsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
            srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        arma::vec stsy = slsy + srsy;
        logger.log("Arma::vec initialized");
        
        
        if ((sl.n0>4) && (sr.n0>4)) {
            lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
            lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
            lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
            /*for (size_t k=0; k<di.p_y; ++k) {
                lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
            }*/
            /*lill = loglike(sl.n, sl.sy, pi.sigma, pi.tau);
            lilr = loglike(sr.n, sr.sy, pi.sigma, pi.tau);
            lilt = loglike(sl.n+sr.n, sl.sy+sr.sy, pi.sigma, pi.tau);*/
            
            alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx); // prior*proposal prob is this odd?, yes, after takeing exp this produces the difference in likelihood
            alpha2 = alpha1*exp(lill+lilr-lilt);//prior*proposal*likelihood
            alpha =std::min(1.0, alpha2);
        } else {
            alpha = 0.0;
        }
        
        //--------------
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mul(di.p_y), mur(di.p_y);//double mul,mur;
        //mul.resize(di.p_y); mur.resize(di.p_y);
        
        if (gen.uniform()<alpha) { //accept birth
            logger.log("Accepting Birth");
            // draw mul
            arma::mat postSigl = (sl.n0 * Covsig.i() + Covprior.i()).i();
            arma::mat postSigr = (sr.n0 * Covsig.i() + Covprior.i()).i();
            //logger.log("Arma mat postsig computed");
            arma::vec arma_mul = Covsig.i() * postSigl * slsy;
            arma::vec arma_mur = Covsig.i() * postSigr * srsy;
            //logger.log("Arma vec mu sampled");
            /*Rcpp::Rcout << "postsigl: " << postSigl.n_rows << " rows, " << postSigl.n_cols << "cols" << endl;
            Rcpp::Rcout << "postsigr: " << postSigr.n_rows << " rows, " << postSigr.n_cols << "cols" << endl;
            Rcpp::Rcout << "arma_mul: " << arma_mul.size() << endl;
            Rcpp::Rcout << "arma_mur: " << arma_mur.size() << endl;*/
            arma::mat samplel = gen.mvnorm(arma_mul, postSigl);
            arma::mat sampler = gen.mvnorm(arma_mur, postSigr);
            //logger.log("Arma mat sampled from mvtnorm");
            
            
            
            for (size_t k=0; k<di.p_y; ++k) {
                /*
                a = 1.0/(pi.tau[k]*pi.tau[k]); //1/tau^2
                s2 = pi.sigma[k]*pi.sigma[k]; //sigma^2
                //left mean
                yb = sl.sy[k]/sl.n[k];
                b = sl.n[k]/s2;
                mul[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                //draw mur
                yb = sr.sy[k]/sr.n[k];
                b = sr.n[k]/s2;
                mur[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                mul[k] = samplel(k);
                mur[k] = sampler(k);
            }
            
            //do birth
            x.birth(nx->nid(), v, c, mul, mur);
            ivcnt[v] += 1;
            return true;
        } else {
            logger.log("Rejecting Birth");
            return false;
        }
    } else { // if not do birth, do death
        logger.log("Attempting Death:");
        
        //draw proposal
        // draw a nog, any nog is possible
        tree::npv nognodes;
        x.getnogs(nognodes);
        size_t ni = floor(gen.uniform()*nognodes.size());
        tree::tree_p nx = nognodes[ni];
        
        // prepare for MH ratio
        
        double PGny; //prob the nog node grows;
        size_t dny = nx->depth();
        PGny = pi.alpha/pow(1.0+dny, pi.beta);
        
        double PGlx = pgrow(nx->getl(), xi, pi);
        double PGrx = pgrow(nx->getr(), xi, pi);
        
        double PBy; // prob of birth move at y
        if (!(nx->p)) { // is the nog nx the top node
            PBy = 1.0;
        } else {
            PBy = pi.pb;
        }
        
        double Pboty; // prob of choosing the nog as bot to split on??
        int ngood = goodbots.size();
        if (cansplit(nx->getl(), xi)) --ngood; // if can split at left child, loose this one;
        if (cansplit(nx->getr(), xi)) --ngood;
        ++ngood; // if we can split can nx
        Pboty = 1.0/ngood;
        
        double PDx = 1.0-PBx; // prob pf death step at x(state);
        if ((PDx>0.)&(PDx<1.)) {PDx -= 0.5;}
        double Pnogx = 1.0/nognodes.size();
        
        // suff stats
        sinfo sl,sr;
        sl.ysize = di.p_y; sr.ysize = di.p_y;
        sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
#ifdef MPIBART
        MPImastergetsuff(x, nx->getl(),nx->getr(),sl,sr,numslaves);
#else
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di, phi, sl, sr);
        
     /*   Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
#endif
        
        //--------------
        //compute alpha
        double lill = 0.0, lilr = 0.0, lilt = 0.0;
        arma::vec slsy(di.p_y, arma::fill::zeros);
        arma::vec srsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
            srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        arma::vec stsy = slsy + srsy;
        lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
        lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
        lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
        /*for (size_t k=0; k<di.p_y; ++k) {
            lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
            lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
            lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        /*double lill = loglike(sl.n, sl.sy, pi.sigma, pi.tau);
        double lilr = loglike(sr.n, sr.sy, pi.sigma, pi.tau);
        double lilt = loglike(sl.n+sr.n, sl.sy+sr.sy, pi.sigma, pi.tau);*/
        
        double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
        double alpha2 = alpha1*exp(lilt - lill - lilr);
        double alpha = std::min(1.0, alpha2);
        
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mu; mu.resize(di.p_y);
        double n;
        
        if (gen.uniform()<alpha) { //accept death
            logger.log("Accepting Death");
            //draw mu for nog(which will be bot)
            arma::mat postSig = ((sl.n0 + sr.n0)*Covsig.i() + Covprior.i()).i();
            arma::vec arma_mu = Covsig.i() * postSig * stsy;
            arma::mat sample = gen.mvnorm(arma_mu, postSig);
            
            for (size_t k=0; k<di.p_y; ++k) {
                /*
                n = sl.n[k]+sr.n[k];
                a = 1.0/(pi.tau[k]*pi.tau[k]);
                s2 = pi.sigma[k]*pi.sigma[k];
                yb = (sl.sy[k]+sr.sy[k])/n;
                b = n/s2;
                mu[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                mu[k] = sample(k);
            }
            
            // do death;
            ivcnt[nx->getv()] -= 1;
            x.death(nx->nid(), mu);
#ifdef MPIBART
            MPImastersenddeath(nx,mu,numslaves);
#endif
            return true;
        } else {
            logger.log("Rejecting Death");
#ifdef MPIBART
            MPImastersendnobirthdeath(numslaves);
#endif
            return false;
        }
        
    }
}

// with coxph model invovlved
bool bd_lg(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv goodbots; // bot nodes that can split
    double PBx = getpb(x, xi, pi, goodbots);
    
    if (gen.uniform() < PBx) {
        logger.log("Attempting Birth");
        
        // draw proposal
        
        // uniformly draw bottom node: choose node index from goodbots
        size_t ni = floor(gen.uniform()*goodbots.size()); // rounddown
        tree::tree_p nx = goodbots[ni];
        
        // draw variable v, uniformly
        std::vector<size_t> goodvars;
        getgoodvars(nx, xi, goodvars); // get variable this node can split on
        size_t vi = floor(gen.uniform()*goodvars.size());
        size_t v = goodvars[vi];
        
        // draw cutpoint, uniformly
        int L,U;
        L=0; U=xi[v].size()-1;
        nx->region(v, &L, &U);
        size_t c = L + floor(gen.uniform()*(U-L+1));
        // U-L+1 is the number of available split points
        
        //-------------------
        // prepare for Metropolis hastings
        double Pbotx = 1.0/goodbots.size(); // proposal dist/probability of choosing nx;
        size_t dnx = nx->depth();
        double PGnx = pi.alpha/pow(1.0+dnx, pi.beta); // prior prob of growing at nx;
        
        double PGly,PGry; // prior probs of growing at new children
        if (goodvars.size()>1) {
            PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            PGry = PGly;
        } else { // have only one v to work with
            if ((int)(c-1)<L) { // v exhausted in new left child l, new upper limit would be c-1
                PGly = 0.0;
            } else {
                PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
            if (U<(int)(c+1)) { // v exhausted in new right child r, new lower limit would be c+1
                PGry = 0.0;
            } else {
                PGry = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
        }
        
        double PDy; // prob of proposing death at y;
        if (goodbots.size()>1) { // can birth at y because splittable nodes left
            PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
        } else { //nx is the only splittable node
            if ((PGry==0) && (PGly==0)) { //cannot birth at y
                PDy = 1.0;
            } else { // y can birth can either l or r
                PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
            }
        }
        
        double Pnogy; // death prob of choosing the nog node at y
        size_t nnogs = x.nnogs();
        tree::tree_cp nxp = nx->getp();
        if (nxp==0) {
            Pnogy = 1.0;
        } else {
            if (nxp->isnog()) { // is parent is a nog, nnogs same at state x and y
                Pnogy = 1.0/nnogs;
            } else {
                Pnogy = 1.0/(nnogs+1.0);
            }
        } // we need Pnogy because we have to choose a nog to perform death at state y.
        
        //-----------
        //compute suff stats
        sinfo sl,sr;
        getsuffBirth(x, nx, v, c, di.p_y, xi, di, phi, sl, sr);
      /*  Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
        
        sinfo sl_noexp, sr_noexp;
        dinfo di_noexp = di;
        double* r_noexp = new double[di.N];
        for (size_t k=0; k<di.N; ++k) {
            r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
        }
        di_noexp.y = r_noexp;
        di_noexp.p_y = 1;
        getsuffBirth(x, nx, v, c, di_noexp.p_y, xi, di_noexp, di_noexp.delta, sl_noexp, sr_noexp);
     /*   Rcpp::Rcout << "sl_noexp: " << sl_noexp.n0 << endl;
        Rcpp::Rcout << sl_noexp.n << endl;
        Rcpp::Rcout << sl_noexp.sdelta << endl;
        Rcpp::Rcout << sl_noexp.sy << endl;
        Rcpp::Rcout << "sr_noexp: " << sr_noexp.n0 << endl;
        Rcpp::Rcout << sr_noexp.n << endl;
        Rcpp::Rcout << sr_noexp.sdelta << endl;
        Rcpp::Rcout << sr_noexp.sy << endl;*/
        
        //---------
        //compute alpha
        double alpha=0.0, alpha1=0.0, alpha2=0.0;
        double lill=0.0, lilr=0.0, lilt=0.0;
        arma::vec slsy(fmax(1,di.p_y-1), arma::fill::zeros);
        arma::vec srsy(fmax(1,di.p_y-1), arma::fill::zeros);
        if (di.p_y>1) {
            for (size_t k=0; k<di.p_y-1; ++k) {
                slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
                srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
            }
        }
        arma::vec stsy = slsy + srsy;
            
        if ((sl.n0>4) && (sr.n0>4)) {
            if (di.p_y>1) {
                lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
                lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
                lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
                /*
                for (size_t k=0; k<di.p_y-1; ++k) {
                    lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                    lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                    lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
                }*/
            }
            lill += loglikelg(sl.sdelta, sl.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0];
            lilr += loglikelg(sr.sdelta, sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sr_noexp.sy[0];
            lilt += loglikelg(sl.sdelta+sr.sdelta, sl.sy[di.p_y-1]+sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0] + sr_noexp.sy[0];
            
     /*       Rcpp::Rcout << "lill: " << lill << endl;
            Rcpp::Rcout << "lilr: " << lilr << endl;
            Rcpp::Rcout << "lilt: " << lilt << endl;*/
            
            
            alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx); // prior*proposal prob is this odd?, yes, after takeing exp this produces the difference in likelihood
            alpha2 = alpha1*exp(lill+lilr-lilt);//prior*proposal*likelihood
            alpha =std::min(1.0, alpha2);
        } else {
            alpha = 0.0;
        }
        
        //--------------
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mul(di.p_y), mur(di.p_y);//double mul,mur;
        //mul.resize(di.p_y); mur.resize(di.p_y);
        
        if (gen.uniform()<alpha) { //accept birth
            logger.log("Accepting Birth");
            if (di.p_y>1) {
                arma::mat postSigl = (sl.n0*Covsig.i() + Covprior.i()).i();
                arma::mat postSigr = (sr.n0*Covsig.i() + Covprior.i()).i();
                arma::vec arma_mul = Covsig.i() * postSigl * slsy;
                arma::vec arma_mur = Covsig.i() * postSigr * srsy;
                arma::mat samplel = gen.mvnorm(arma_mul, postSigl);
                arma::mat sampler = gen.mvnorm(arma_mur, postSigr);
                // draw mu for normal response
                for (size_t k=0; k<di.p_y-1; ++k) {
                    /*
                    a = 1.0/(pi.tau[k]*pi.tau[k]); //1/tau^2
                    s2 = pi.sigma[k]*pi.sigma[k]; //sigma^2
                    //left mean
                    yb = sl.sy[k]/sl.n[k];
                    b = sl.n[k]/s2;
                    mul[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                    //draw mur
                    yb = sr.sy[k]/sr.n[k];
                    b = sr.n[k]/s2;
                    mur[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                    */
                    mul[k] = samplel(k);
                    mur[k] = sampler(k);
                }
            }
            // draw mu for coxph
            mul[di.p_y-1] = gen.loggamma(sl.sdelta+pi.lg_alpha, 1/(sl.sy[di.p_y-1] + pi.lg_beta));
            mur[di.p_y-1] = gen.loggamma(sr.sdelta+pi.lg_alpha, 1/(sr.sy[di.p_y-1] + pi.lg_beta));
            
            //do birth
            x.birth(nx->nid(), v, c, mul, mur);
            ivcnt[v] += 1;
            delete []  r_noexp;
            return true;
        } else {
            logger.log("Rejecting Birth");
            delete []  r_noexp;
            return false;
        }
        
    } else { // if not do birth, do death
        logger.log("Attempting Death:");
        
        //draw proposal
        // draw a nog, any nog is possible
        tree::npv nognodes;
        x.getnogs(nognodes);
        size_t ni = floor(gen.uniform()*nognodes.size());
        tree::tree_p nx = nognodes[ni];
        
        // prepare for MH ratio
        
        double PGny; //prob the nog node grows;
        size_t dny = nx->depth();
        PGny = pi.alpha/pow(1.0+dny, pi.beta);
        
        double PGlx = pgrow(nx->getl(), xi, pi);
        double PGrx = pgrow(nx->getr(), xi, pi);
        
        double PBy; // prob of birth move at y
        if (!(nx->p)) { // is the nog nx the top node
            PBy = 1.0;
        } else {
            PBy = pi.pb;
        }
        
        double Pboty; // prob of choosing the nog as bot to split on??
        int ngood = goodbots.size();
        if (cansplit(nx->getl(), xi)) --ngood; // if can split at left child, loose this one;
        if (cansplit(nx->getr(), xi)) --ngood;
        ++ngood; // if we can split can nx
        Pboty = 1.0/ngood;
        
        double PDx = 1.0-PBx; // prob pf death step at x(state);
        if ((PDx>0.)&(PDx<1.)) {PDx -= 0.5;}
        double Pnogx = 1.0/nognodes.size();
        
        // suff stats
        sinfo sl,sr;
        sl.ysize = di.p_y; sr.ysize = di.p_y;
        sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
#ifdef MPIBART
        MPImastergetsuff(x, nx->getl(),nx->getr(),sl,sr,numslaves);
#else
        /*for (size_t k=0; k<di.N; ++k) {
            Rcpp::Rcout << phi[k]  << "x" << di.y[k]<< endl;
        }*/
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di, phi, sl, sr);
        
   /*     Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
#endif
        
        sinfo sl_noexp, sr_noexp;
        sl_noexp.ysize = 1; sr_noexp.ysize = 1;
        sl_noexp.sy.resize(1); sr_noexp.sy.resize(1);
        dinfo di_noexp = di;
        double* r_noexp = new double[di.N];
        for (size_t k=0; k<di.N; ++k) {
            r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
        }
        di_noexp.y = r_noexp;
        di_noexp.p_y = 1;
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di_noexp, di.delta, sl_noexp, sr_noexp);
  /*      Rcpp::Rcout << "sl_noexp: " << sl_noexp.n0 << endl;
        Rcpp::Rcout << sl_noexp.n << endl;
        Rcpp::Rcout << sl_noexp.sdelta << endl;
        Rcpp::Rcout << sl_noexp.sy << endl;
        Rcpp::Rcout << "sr_noexp: " << sr_noexp.n0 << endl;
        Rcpp::Rcout << sr_noexp.n << endl;
        Rcpp::Rcout << sr_noexp.sdelta << endl;
        Rcpp::Rcout << sr_noexp.sy << endl;*/
        
        //--------------
        //compute alpha
        double lill = 0.0, lilr = 0.0, lilt = 0.0;
        arma::vec slsy(fmax(1,di.p_y-1), arma::fill::zeros);
        arma::vec srsy(fmax(1,di.p_y-1), arma::fill::zeros);
        if (di.p_y>1) {
            for (size_t k=0; k<di.p_y-1; ++k) {
                slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
                srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
            }
        }
        arma::vec stsy = slsy + srsy;
        if (di.p_y>1) {
            lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
            lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
            lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
            }
             */
        }
        lill += loglikelg(sl.sdelta, sl.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0];
        lilr += loglikelg(sr.sdelta, sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sr_noexp.sy[0];
        lilt += loglikelg(sl.sdelta+sr.sdelta, sl.sy[di.p_y-1]+sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0] + sr_noexp.sy[0];
        
    /*    Rcpp::Rcout << "lill: " << lill << endl;
        Rcpp::Rcout << "lilr: " << lilr << endl;
        Rcpp::Rcout << "lilt: " << lilt << endl;*/
        
        double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
        double alpha2 = alpha1*exp(lilt - lill - lilr);
        double alpha = std::min(1.0, alpha2);
        
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mu; mu.resize(di.p_y);
        double n;
        
        if (gen.uniform()<alpha) { //accept death
            logger.log("Accepting Death");
            if (di.p_y>1) {
                //draw mu for nog(which will be bot)
                arma::mat postSig = ((sl.n0 + sr.n0)*Covsig.i() + Covprior.i()).i();
                arma::vec arma_mu = Covsig.i() * postSig * stsy;
                arma::mat sample = gen.mvnorm(arma_mu, postSig);
                for (size_t k=0; k<di.p_y-1; ++k) {
                    /*
                    n = sl.n[k]+sr.n[k];
                    a = 1.0/(pi.tau[k]*pi.tau[k]);
                    s2 = pi.sigma[k]*pi.sigma[k];
                    yb = (sl.sy[k]+sr.sy[k])/n;
                    b = n/s2;
                    mu[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                    mu[k] = sample(k);
                }
            }
            mu[di.p_y-1] = gen.loggamma(sl.sdelta+sr.sdelta+pi.lg_alpha, 1/(sl.sy[di.p_y-1]+sr.sy[di.p_y-1]+pi.lg_beta));
            
            // do death;
            ivcnt[nx->getv()] -= 1;
            x.death(nx->nid(), mu);
#ifdef MPIBART
            MPImastersenddeath(nx,mu,numslaves);
#endif
            return true;
        } else {
            logger.log("Rejecting Death");
#ifdef MPIBART
            MPImastersendnobirthdeath(numslaves);
#endif
            return false;
        }
        delete [] r_noexp;
    }
    
}



bool bd_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers)
{
    tree::npv goodbots; // bot nodes that can split
    double PBx = getpb(x, xi, pi, goodbots);
    
    if (gen.uniform() < PBx) {
        logger.log("Attempting Birth");
        
        // draw proposal
        
        // uniformly draw bottom node: choose node index from goodbots
        size_t ni = floor(gen.uniform()*goodbots.size()); // rounddown
        tree::tree_p nx = goodbots[ni];
        
        // draw variable v, uniformly
        std::vector<size_t> goodvars;
        getgoodvars(nx, xi, goodvars); // get variable this node can split on
        
        
        size_t vi = floor(gen.uniform()*goodvars.size());
        size_t v = goodvars[vi];
        
        if (vs) {
            /*
            vi = sample_class(probs, gen);//floor(gen.uniform()*goodvars.size());
            while (std::find(goodvars.begin(), goodvars.end(), vi)==goodvars.end()) {
                    vi = sample_class(probs, gen);
            }
            v = vi;*/
            vi = hypers.SampleVar(gen);
            while (std::find(goodvars.begin(), goodvars.end(), vi)==goodvars.end()) {
                    vi = hypers.SampleVar(gen);
            }
            v = vi;
            //hypers.counts[v] += 1;
        }
        
        
      /*  cout << goodvars.size() << "vars at choice" << endl;
        cout << vi  << "th goodvar is chosen" << endl;
        cout << v << "th predictor is chosen " << endl;*/
        
        // draw cutpoint, uniformly
        int L,U;
        L=0; U=xi[v].size()-1;
        nx->region(v, &L, &U);
        size_t c = L + floor(gen.uniform()*(U-L+1));
        // U-L+1 is the number of available split points
        
        //-------------------
        // prepare for Metropolis hastings
        double Pbotx = 1.0/goodbots.size(); // proposal dist/probability of choosing nx;
        size_t dnx = nx->depth();
        double PGnx = pi.alpha/pow(1.0+dnx, pi.beta); // prior prob of growing at nx;
        
        double PGly,PGry; // prior probs of growing at new children
        if (goodvars.size()>1) {
            PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            PGry = PGly;
        } else { // have only one v to work with
            if ((int)(c-1)<L) { // v exhausted in new left child l, new upper limit would be c-1
                PGly = 0.0;
            } else {
                PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
            if (U<(int)(c+1)) { // v exhausted in new right child r, new lower limit would be c+1
                PGry = 0.0;
            } else {
                PGry = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
        }
        
        double PDy; // prob of proposing death at y;
        if (goodbots.size()>1) { // can birth at y because splittable nodes left
            PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
        } else { //nx is the only splittable node
            if ((PGry==0) && (PGly==0)) { //cannot birth at y
                PDy = 1.0;
            } else { // y can birth can either l or r
                PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
            }
        }
        
        double Pnogy; // death prob of choosing the nog node at y
        size_t nnogs = x.nnogs();
        tree::tree_cp nxp = nx->getp();
        if (nxp==0) {
            Pnogy = 1.0;
        } else {
            if (nxp->isnog()) { // is parent is a nog, nnogs same at state x and y
                Pnogy = 1.0/nnogs;
            } else {
                Pnogy = 1.0/(nnogs+1.0);
            }
        } // we need Pnogy because we have to choose a nog to perform death at state y.
        
        //-----------
        //compute suff stats
        sinfo sl,sr;
        //sl.ysize = di.p_y; sr.ysize = di.p_y;
        //sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        //sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
        getsuffBirth(x, nx, v, c, di.p_y, xi, di, phi, sl, sr);
        
        /*
        Rcpp::Rcout << "left" << sl.n0  << " " << sl.ysize << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sy << endl;
        
        Rcpp::Rcout << "right" << sr.n0  <<  sr.ysize << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sy << endl;*/
        //---------
        //compute alpha
        double alpha=0.0, alpha1=0.0, alpha2=0.0;
        double lill=0.0, lilr=0.0, lilt=0.0;
        arma::vec slsy(di.p_y, arma::fill::zeros);
        arma::vec srsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
            srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        arma::vec stsy = slsy + srsy;
        logger.log("Arma::vec initialized");
        
        
        if ((sl.n0>4) && (sr.n0>4)) {
            lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
            lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
            lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
            /*for (size_t k=0; k<di.p_y; ++k) {
                lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
            }*/
            /*lill = loglike(sl.n, sl.sy, pi.sigma, pi.tau);
            lilr = loglike(sr.n, sr.sy, pi.sigma, pi.tau);
            lilt = loglike(sl.n+sr.n, sl.sy+sr.sy, pi.sigma, pi.tau);*/
            
            alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx); // prior*proposal prob is this odd?, yes, after takeing exp this produces the difference in likelihood
            alpha2 = alpha1*exp(lill+lilr-lilt);//prior*proposal*likelihood
            alpha =std::min(1.0, alpha2);
        } else {
            alpha = 0.0;
        }
        
        //--------------
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mul(di.p_y), mur(di.p_y);//double mul,mur;
        //mul.resize(di.p_y); mur.resize(di.p_y);
        
        if (gen.uniform()<alpha) { //accept birth
            logger.log("Accepting Birth");
            // draw mul
            arma::mat postSigl = (sl.n0 * Covsig.i() + Covprior.i()).i();
            arma::mat postSigr = (sr.n0 * Covsig.i() + Covprior.i()).i();
            //logger.log("Arma mat postsig computed");
            arma::vec arma_mul = Covsig.i() * postSigl * slsy;
            arma::vec arma_mur = Covsig.i() * postSigr * srsy;
            //logger.log("Arma vec mu sampled");
            /*Rcpp::Rcout << "postsigl: " << postSigl.n_rows << " rows, " << postSigl.n_cols << "cols" << endl;
            Rcpp::Rcout << "postsigr: " << postSigr.n_rows << " rows, " << postSigr.n_cols << "cols" << endl;
            Rcpp::Rcout << "arma_mul: " << arma_mul.size() << endl;
            Rcpp::Rcout << "arma_mur: " << arma_mur.size() << endl;*/
            arma::mat samplel = gen.mvnorm(arma_mul, postSigl);
            arma::mat sampler = gen.mvnorm(arma_mur, postSigr);
            //logger.log("Arma mat sampled from mvtnorm");
            
            
            
            for (size_t k=0; k<di.p_y; ++k) {
                /*
                a = 1.0/(pi.tau[k]*pi.tau[k]); //1/tau^2
                s2 = pi.sigma[k]*pi.sigma[k]; //sigma^2
                //left mean
                yb = sl.sy[k]/sl.n[k];
                b = sl.n[k]/s2;
                mul[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                //draw mur
                yb = sr.sy[k]/sr.n[k];
                b = sr.n[k]/s2;
                mur[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                mul[k] = samplel(k);
                mur[k] = sampler(k);
            }
            
            //do birth
            x.birth(nx->nid(), v, c, mul, mur);
            ivcnt[v] += 1;
            hypers.counts[v] += 1;
            return true;
        } else {
            logger.log("Rejecting Birth");
            return false;
        }
    } else { // if not do birth, do death
        logger.log("Attempting Death:");
        
        //draw proposal
        // draw a nog, any nog is possible
        tree::npv nognodes;
        x.getnogs(nognodes);
        size_t ni = floor(gen.uniform()*nognodes.size());
        tree::tree_p nx = nognodes[ni];
        
        // prepare for MH ratio
        
        double PGny; //prob the nog node grows;
        size_t dny = nx->depth();
        PGny = pi.alpha/pow(1.0+dny, pi.beta);
        
        double PGlx = pgrow(nx->getl(), xi, pi);
        double PGrx = pgrow(nx->getr(), xi, pi);
        
        double PBy; // prob of birth move at y
        if (!(nx->p)) { // is the nog nx the top node
            PBy = 1.0;
        } else {
            PBy = pi.pb;
        }
        
        double Pboty; // prob of choosing the nog as bot to split on??
        int ngood = goodbots.size();
        if (cansplit(nx->getl(), xi)) --ngood; // if can split at left child, loose this one;
        if (cansplit(nx->getr(), xi)) --ngood;
        ++ngood; // if we can split can nx
        Pboty = 1.0/ngood;
        
        double PDx = 1.0-PBx; // prob pf death step at x(state);
        if ((PDx>0.)&(PDx<1.)) {PDx -= 0.5;}
        double Pnogx = 1.0/nognodes.size();
        
        // suff stats
        sinfo sl,sr;
        sl.ysize = di.p_y; sr.ysize = di.p_y;
        sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
#ifdef MPIBART
        MPImastergetsuff(x, nx->getl(),nx->getr(),sl,sr,numslaves);
#else
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di, phi, sl, sr);
        
     /*   Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
#endif
        
        //--------------
        //compute alpha
        double lill = 0.0, lilr = 0.0, lilt = 0.0;
        arma::vec slsy(di.p_y, arma::fill::zeros);
        arma::vec srsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
            srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        arma::vec stsy = slsy + srsy;
        lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
        lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
        lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
        /*for (size_t k=0; k<di.p_y; ++k) {
            lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
            lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
            lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
        }*/
        /*double lill = loglike(sl.n, sl.sy, pi.sigma, pi.tau);
        double lilr = loglike(sr.n, sr.sy, pi.sigma, pi.tau);
        double lilt = loglike(sl.n+sr.n, sl.sy+sr.sy, pi.sigma, pi.tau);*/
        
        double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
        double alpha2 = alpha1*exp(lilt - lill - lilr);
        double alpha = std::min(1.0, alpha2);
        
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mu; mu.resize(di.p_y);
        double n;
        
        if (gen.uniform()<alpha) { //accept death
            logger.log("Accepting Death");
            //draw mu for nog(which will be bot)
            arma::mat postSig = ((sl.n0 + sr.n0)*Covsig.i() + Covprior.i()).i();
            arma::vec arma_mu = Covsig.i() * postSig * stsy;
            arma::mat sample = gen.mvnorm(arma_mu, postSig);
            
            for (size_t k=0; k<di.p_y; ++k) {
                /*
                n = sl.n[k]+sr.n[k];
                a = 1.0/(pi.tau[k]*pi.tau[k]);
                s2 = pi.sigma[k]*pi.sigma[k];
                yb = (sl.sy[k]+sr.sy[k])/n;
                b = n/s2;
                mu[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                mu[k] = sample(k);
            }
            
            // do death;
            ivcnt[nx->getv()] -= 1;
            SubtractTreeCounts(hypers, nx);
            x.death(nx->nid(), mu);
            
#ifdef MPIBART
            MPImastersenddeath(nx,mu,numslaves);
#endif
            return true;
        } else {
            logger.log("Rejecting Death");
#ifdef MPIBART
            MPImastersendnobirthdeath(numslaves);
#endif
            return false;
        }
        
    }
}






bool bd_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers)
{
    tree::npv goodbots; // bot nodes that can split
    double PBx = getpb(x, xi, pi, goodbots);
    
    if (gen.uniform() < PBx) {
        logger.log("Attempting Birth");
        
        // draw proposal
        
        // uniformly draw bottom node: choose node index from goodbots
        size_t ni = floor(gen.uniform()*goodbots.size()); // rounddown
        tree::tree_p nx = goodbots[ni];
        
        // draw variable v, uniformly
        std::vector<size_t> goodvars;
        getgoodvars(nx, xi, goodvars); // get variable this node can split on
        size_t vi = floor(gen.uniform()*goodvars.size());
        size_t v = goodvars[vi];
        
        if (vs) {
            /*
            vi = sample_class(probs, gen);//floor(gen.uniform()*goodvars.size());
            while (std::find(goodvars.begin(), goodvars.end(), vi)==goodvars.end()) {
                    vi = sample_class(probs, gen);
            }*/
            vi = hypers.SampleVar(gen);
            while (std::find(goodvars.begin(), goodvars.end(), vi)==goodvars.end()) {
                    vi = hypers.SampleVar(gen);
            }
            v = vi;
        }
        
        // draw cutpoint, uniformly
        int L,U;
        L=0; U=xi[v].size()-1;
        nx->region(v, &L, &U);
        size_t c = L + floor(gen.uniform()*(U-L+1));
        // U-L+1 is the number of available split points
        
        //-------------------
        // prepare for Metropolis hastings
        double Pbotx = 1.0/goodbots.size(); // proposal dist/probability of choosing nx;
        size_t dnx = nx->depth();
        double PGnx = pi.alpha/pow(1.0+dnx, pi.beta); // prior prob of growing at nx;
        
        double PGly,PGry; // prior probs of growing at new children
        if (goodvars.size()>1) {
            PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            PGry = PGly;
        } else { // have only one v to work with
            if ((int)(c-1)<L) { // v exhausted in new left child l, new upper limit would be c-1
                PGly = 0.0;
            } else {
                PGly = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
            if (U<(int)(c+1)) { // v exhausted in new right child r, new lower limit would be c+1
                PGry = 0.0;
            } else {
                PGry = pi.alpha/pow(1.0+dnx+1.0, pi.beta);
            }
        }
        
        double PDy; // prob of proposing death at y;
        if (goodbots.size()>1) { // can birth at y because splittable nodes left
            PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
        } else { //nx is the only splittable node
            if ((PGry==0) && (PGly==0)) { //cannot birth at y
                PDy = 1.0;
            } else { // y can birth can either l or r
                PDy = pi.pbd - pi.pb;//1.0 - pi.pb;
            }
        }
        
        double Pnogy; // death prob of choosing the nog node at y
        size_t nnogs = x.nnogs();
        tree::tree_cp nxp = nx->getp();
        if (nxp==0) {
            Pnogy = 1.0;
        } else {
            if (nxp->isnog()) { // is parent is a nog, nnogs same at state x and y
                Pnogy = 1.0/nnogs;
            } else {
                Pnogy = 1.0/(nnogs+1.0);
            }
        } // we need Pnogy because we have to choose a nog to perform death at state y.
        
        //-----------
        //compute suff stats
        sinfo sl,sr;
        getsuffBirth(x, nx, v, c, di.p_y, xi, di, phi, sl, sr);
      /*  Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
        
        sinfo sl_noexp, sr_noexp;
        dinfo di_noexp = di;
        double* r_noexp = new double[di.N];
        for (size_t k=0; k<di.N; ++k) {
            r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
        }
        di_noexp.y = r_noexp;
        di_noexp.p_y = 1;
        getsuffBirth(x, nx, v, c, di_noexp.p_y, xi, di_noexp, di_noexp.delta, sl_noexp, sr_noexp);
     /*   Rcpp::Rcout << "sl_noexp: " << sl_noexp.n0 << endl;
        Rcpp::Rcout << sl_noexp.n << endl;
        Rcpp::Rcout << sl_noexp.sdelta << endl;
        Rcpp::Rcout << sl_noexp.sy << endl;
        Rcpp::Rcout << "sr_noexp: " << sr_noexp.n0 << endl;
        Rcpp::Rcout << sr_noexp.n << endl;
        Rcpp::Rcout << sr_noexp.sdelta << endl;
        Rcpp::Rcout << sr_noexp.sy << endl;*/
        
        //---------
        //compute alpha
        double alpha=0.0, alpha1=0.0, alpha2=0.0;
        double lill=0.0, lilr=0.0, lilt=0.0;
        arma::vec slsy(fmax(1,di.p_y-1), arma::fill::zeros);
        arma::vec srsy(fmax(1,di.p_y-1), arma::fill::zeros);
        if (di.p_y>1) {
            for (size_t k=0; k<di.p_y-1; ++k) {
                slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
                srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
            }
        }
        arma::vec stsy = slsy + srsy;
            
        if ((sl.n0>4) && (sr.n0>4)) {
            if (di.p_y>1) {
                lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
                lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
                lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
                /*
                for (size_t k=0; k<di.p_y-1; ++k) {
                    lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                    lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                    lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
                }*/
            }
            lill += loglikelg(sl.sdelta, sl.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0];
            lilr += loglikelg(sr.sdelta, sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sr_noexp.sy[0];
            lilt += loglikelg(sl.sdelta+sr.sdelta, sl.sy[di.p_y-1]+sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0] + sr_noexp.sy[0];
            
     /*       Rcpp::Rcout << "lill: " << lill << endl;
            Rcpp::Rcout << "lilr: " << lilr << endl;
            Rcpp::Rcout << "lilt: " << lilt << endl;*/
            
            
            alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx); // prior*proposal prob is this odd?, yes, after takeing exp this produces the difference in likelihood
            alpha2 = alpha1*exp(lill+lilr-lilt);//prior*proposal*likelihood
            alpha =std::min(1.0, alpha2);
        } else {
            alpha = 0.0;
        }
        
        //--------------
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mul(di.p_y), mur(di.p_y);//double mul,mur;
        //mul.resize(di.p_y); mur.resize(di.p_y);
        
        if (gen.uniform()<alpha) { //accept birth
            logger.log("Accepting Birth");
            if (di.p_y>1) {
                arma::mat postSigl = (sl.n0*Covsig.i() + Covprior.i()).i();
                arma::mat postSigr = (sr.n0*Covsig.i() + Covprior.i()).i();
                arma::vec arma_mul = Covsig.i() * postSigl * slsy;
                arma::vec arma_mur = Covsig.i() * postSigr * srsy;
                arma::mat samplel = gen.mvnorm(arma_mul, postSigl);
                arma::mat sampler = gen.mvnorm(arma_mur, postSigr);
                // draw mu for normal response
                for (size_t k=0; k<di.p_y-1; ++k) {
                    /*
                    a = 1.0/(pi.tau[k]*pi.tau[k]); //1/tau^2
                    s2 = pi.sigma[k]*pi.sigma[k]; //sigma^2
                    //left mean
                    yb = sl.sy[k]/sl.n[k];
                    b = sl.n[k]/s2;
                    mul[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                    //draw mur
                    yb = sr.sy[k]/sr.n[k];
                    b = sr.n[k]/s2;
                    mur[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);
                    */
                    mul[k] = samplel(k);
                    mur[k] = sampler(k);
                }
            }
            // draw mu for coxph
            mul[di.p_y-1] = gen.loggamma(sl.sdelta+pi.lg_alpha, 1/(sl.sy[di.p_y-1] + pi.lg_beta));
            mur[di.p_y-1] = gen.loggamma(sr.sdelta+pi.lg_alpha, 1/(sr.sy[di.p_y-1] + pi.lg_beta));
            
            //do birth
            x.birth(nx->nid(), v, c, mul, mur);
            ivcnt[v] += 1;
            hypers.counts[v] += 1;
            delete []  r_noexp;
            return true;
        } else {
            logger.log("Rejecting Birth");
            delete []  r_noexp;
            return false;
        }
        
    } else { // if not do birth, do death
        logger.log("Attempting Death:");
        
        //draw proposal
        // draw a nog, any nog is possible
        tree::npv nognodes;
        x.getnogs(nognodes);
        size_t ni = floor(gen.uniform()*nognodes.size());
        tree::tree_p nx = nognodes[ni];
        
        // prepare for MH ratio
        
        double PGny; //prob the nog node grows;
        size_t dny = nx->depth();
        PGny = pi.alpha/pow(1.0+dny, pi.beta);
        
        double PGlx = pgrow(nx->getl(), xi, pi);
        double PGrx = pgrow(nx->getr(), xi, pi);
        
        double PBy; // prob of birth move at y
        if (!(nx->p)) { // is the nog nx the top node
            PBy = 1.0;
        } else {
            PBy = pi.pb;
        }
        
        double Pboty; // prob of choosing the nog as bot to split on??
        int ngood = goodbots.size();
        if (cansplit(nx->getl(), xi)) --ngood; // if can split at left child, loose this one;
        if (cansplit(nx->getr(), xi)) --ngood;
        ++ngood; // if we can split can nx
        Pboty = 1.0/ngood;
        
        double PDx = 1.0-PBx; // prob pf death step at x(state);
        if ((PDx>0.)&(PDx<1.)) {PDx -= 0.5;}
        double Pnogx = 1.0/nognodes.size();
        
        // suff stats
        sinfo sl,sr;
        sl.ysize = di.p_y; sr.ysize = di.p_y;
        sl.sy.resize(di.p_y); sl.n.resize(di.p_y);
        sr.sy.resize(di.p_y); sr.n.resize(di.p_y);
#ifdef MPIBART
        MPImastergetsuff(x, nx->getl(),nx->getr(),sl,sr,numslaves);
#else
        /*for (size_t k=0; k<di.N; ++k) {
            Rcpp::Rcout << phi[k]  << "x" << di.y[k]<< endl;
        }*/
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di, phi, sl, sr);
        
   /*     Rcpp::Rcout << "sl: " << sl.n0 << endl;
        Rcpp::Rcout << sl.n << endl;
        Rcpp::Rcout << sl.sdelta << endl;
        Rcpp::Rcout << sl.sy << endl;
        Rcpp::Rcout << "sr: " << sr.n0 << endl;
        Rcpp::Rcout << sr.n << endl;
        Rcpp::Rcout << sr.sdelta << endl;
        Rcpp::Rcout << sr.sy << endl;*/
#endif
        
        sinfo sl_noexp, sr_noexp;
        sl_noexp.ysize = 1; sr_noexp.ysize = 1;
        sl_noexp.sy.resize(1); sr_noexp.sy.resize(1);
        dinfo di_noexp = di;
        double* r_noexp = new double[di.N];
        for (size_t k=0; k<di.N; ++k) {
            r_noexp[k] = log(di.y[(k+1)*di.p_y-1]);
        }
        di_noexp.y = r_noexp;
        di_noexp.p_y = 1;
        getsuffDeath(x, nx->getl(), nx->getr(), xi, di_noexp, di.delta, sl_noexp, sr_noexp);
  /*      Rcpp::Rcout << "sl_noexp: " << sl_noexp.n0 << endl;
        Rcpp::Rcout << sl_noexp.n << endl;
        Rcpp::Rcout << sl_noexp.sdelta << endl;
        Rcpp::Rcout << sl_noexp.sy << endl;
        Rcpp::Rcout << "sr_noexp: " << sr_noexp.n0 << endl;
        Rcpp::Rcout << sr_noexp.n << endl;
        Rcpp::Rcout << sr_noexp.sdelta << endl;
        Rcpp::Rcout << sr_noexp.sy << endl;*/
        
        //--------------
        //compute alpha
        double lill = 0.0, lilr = 0.0, lilt = 0.0;
        arma::vec slsy(fmax(1,di.p_y-1), arma::fill::zeros);
        arma::vec srsy(fmax(1,di.p_y-1), arma::fill::zeros);
        if (di.p_y>1) {
            for (size_t k=0; k<di.p_y-1; ++k) {
                slsy(k) = sl.sy[k]*pi.sigma[k]*pi.sigma[k];
                srsy(k) = sr.sy[k]*pi.sigma[k]*pi.sigma[k];
            }
        }
        arma::vec stsy = slsy + srsy;
        if (di.p_y>1) {
            lill += loglike_mvn(sl.n0, slsy, Covsig, Covprior);
            lilr += loglike_mvn(sr.n0, srsy, Covsig, Covprior);
            lilt += loglike_mvn(sl.n0+sr.n0, stsy, Covsig, Covprior);
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                lill += loglike(sl.n[k], sl.sy[k], pi.sigma[k], pi.tau[k]);
                lilr += loglike(sr.n[k], sr.sy[k], pi.sigma[k], pi.tau[k]);
                lilt += loglike(sl.n[k]+sr.n[k], sl.sy[k]+sr.sy[k], pi.sigma[k], pi.tau[k]);
            }
             */
        }
        lill += loglikelg(sl.sdelta, sl.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0];
        lilr += loglikelg(sr.sdelta, sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sr_noexp.sy[0];
        lilt += loglikelg(sl.sdelta+sr.sdelta, sl.sy[di.p_y-1]+sr.sy[di.p_y-1], pi.lg_alpha, pi.lg_beta) + sl_noexp.sy[0] + sr_noexp.sy[0];
        
    /*    Rcpp::Rcout << "lill: " << lill << endl;
        Rcpp::Rcout << "lilr: " << lilr << endl;
        Rcpp::Rcout << "lilt: " << lilt << endl;*/
        
        double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
        double alpha2 = alpha1*exp(lilt - lill - lilr);
        double alpha = std::min(1.0, alpha2);
        
        // finally MH
        double a,b,s2,yb;
        std::vector<double> mu; mu.resize(di.p_y);
        double n;
        
        if (gen.uniform()<alpha) { //accept death
            logger.log("Accepting Death");
            if (di.p_y>1) {
                //draw mu for nog(which will be bot)
                arma::mat postSig = ((sl.n0 + sr.n0)*Covsig.i() + Covprior.i()).i();
                arma::vec arma_mu = Covsig.i() * postSig * stsy;
                arma::mat sample = gen.mvnorm(arma_mu, postSig);
                for (size_t k=0; k<di.p_y-1; ++k) {
                    /*
                    n = sl.n[k]+sr.n[k];
                    a = 1.0/(pi.tau[k]*pi.tau[k]);
                    s2 = pi.sigma[k]*pi.sigma[k];
                    yb = (sl.sy[k]+sr.sy[k])/n;
                    b = n/s2;
                    mu[k] = b*yb/(a+b) + gen.normal()/sqrt(a+b);*/
                    mu[k] = sample(k);
                }
            }
            mu[di.p_y-1] = gen.loggamma(sl.sdelta+sr.sdelta+pi.lg_alpha, 1/(sl.sy[di.p_y-1]+sr.sy[di.p_y-1]+pi.lg_beta));
            
            // do death;
            ivcnt[nx->getv()] -= 1;
            SubtractTreeCounts(hypers, nx);
            x.death(nx->nid(), mu);
#ifdef MPIBART
            MPImastersenddeath(nx,mu,numslaves);
#endif
            return true;
        } else {
            logger.log("Rejecting Death");
#ifdef MPIBART
            MPImastersendnobirthdeath(numslaves);
#endif
            return false;
        }
        delete [] r_noexp;
    }
    
}
