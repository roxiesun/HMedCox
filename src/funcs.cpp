// [[Rcpp::depends(RcppParallel)]]
#include <cmath>
#include "funcs.h"
#include <map>
#include <RcppParallel.h>
#ifdef MPIBART
#include "mpi.h"
#endif

using Rcpp::Rcout;
using namespace RcppParallel;
using Rcpp::stop;

// normal density
double pn(double x, double m, double v)
{
    double dif = x-m;
    return exp(-0.5*dif*dif/v)/sqrt(2*PI*v);
}

// discrete distribution draw
int rdisc(double *p, RNG& gen)
{
    double sum;
    double u = gen.uniform();
    
    int i = 0;
    sum = p[0];
    while (sum < u) {
        i += 1;
        sum += p[i];
    }
    return i;
}

// evaluate tree on grid xinfo, and write
void grm(tree&tr, xinfo& xi, std::ostream& os)
{
    size_t p = xi.size();
    //check the dim of xi
    if (p!=2) {
        Rcout << "error in grm: p!=2\n";
        return;
    }
    size_t n1 = xi[0].size();
    size_t n2 = xi[1].size();
    tree::tree_cp bp;
    double *x = new double[2]; // array of two elements
    for (size_t i=0; i!=n1; i++) {
        for (size_t j=0; j!=n2; j++) {
            x[0] = xi[0][i];
            x[1] = xi[1][j];
            bp = tr.bot(x, xi);
            os << x[0] << " " << x[1] << " " << bp->getmu() << " " << bp->nid() << endl;
        }
    }
    delete[] x;
}

// check whether a node has variables it can split on
bool cansplit(tree::tree_p n, xinfo& xi)
{
    int L, U;
    bool v_found = false;
    size_t v = 0;
    while (!v_found && (v < xi.size())) {
        L = 0; U = xi[v].size() - 1;
        n->region(v, &L, &U);
        if (U>=L) v_found = true;
        v++;
    }
    return v_found;
}

// compute prob birth, goodbots save all bottom nodes that can be further splitted
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots)
{
    double pb;
    tree::npv bnv; //all bottom nodes
    t.getbots(bnv);
    for (size_t i=0; i!=bnv.size(); i++) {
        if (cansplit(bnv[i], xi))
            goodbots.push_back(bnv[i]);
    }
    if (goodbots.size()==0) {
        pb = 0.0;
    } else {
        if (t.treesize()==1) pb = 1.0;
        else pb = pi.pb;
    }
    return pb;
}

// find variables the node n can split on and store in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi, std::vector<size_t>& goodvars)
{
    int L,U;
    for (size_t v=0; v!=xi.size(); v++) {
        L=0; U=xi[v].size()-1;
        n->region(v, &L, &U);
        if (U>=L) goodvars.push_back(v);
    }
}

// find variables the INTERNAL node n can split on and store in goodvars
void getinternalvars(tree::tree_p n, xinfo& xi, std::vector<size_t>& goodvars)
{
    int L,U;
    for (size_t v=0; v!=xi.size(); v++) {
        L=0; U=xi[v].size()-1;
        getpertLU(n, v, xi, &L, &U);
        if (U>=L) goodvars.push_back(v);
    }
}

// number of avaible cut points for node n variable var
int getnumcuts(tree::tree_p n, xinfo& xi, size_t var)
{
    int L,U;
    
    getpertLU(n, var, xi, &L, &U);
    return std::max(0, U-L+1);
}

// get prob a node grows, 0 if no good vars, a/(1+d)^b else
double pgrow(tree::tree_p n, xinfo&xi, pinfo& pi)
{
    if (cansplit(n, xi)) {
        return pi.alpha/pow(1.0+n->depth(), pi.beta);
    } else {
        return 0.0;
    }
}

//calibart
// get the L,U values for a node in the tree GIVEN the tree structure below and above that node
void getLU(tree::tree_p pertnode, xinfo& xi, int* L, int* U)
{
    tree::tree_p l,r;
    
    *L=0;
    *U=xi[pertnode->getv()].size()-1;
    l=pertnode->getl();
    r=pertnode->getr();
    
    bool usel, user;
    usel = l->nuse(pertnode->getv());
    user = r->nuse(pertnode->getv());
    if (usel && user) {
        l->region_l(pertnode->getv(), L);
        r->region_u(pertnode->getv(), U);
    }
    else if (usel)
    {
        pertnode->region(pertnode->getv(), L, U);
        l->region_l(pertnode->getv(), L);
    }
    else
    {
        pertnode->region(pertnode->getv(), L, U);
        r->region_u(pertnode->getv(), U);
    }
}

/* Note: region is to search upwards based on current node, region_l is to search downwards knowing that the tree can be further splitted by rule v<=?, region_r is to search downwards and knowing that the tree can be further splitted by rule v>? */

// if the var is prescribed, given the nodes above and below it, what is the availale range for this node with this prescribes var
void getpertLU(tree::tree_p pertnode, size_t pertvar, xinfo& xi, int* L, int* U)
{
    *L=0;
    *U=xi[pertvar].size()-1;
    
    bool usel,user;
    usel=pertnode->l->nuse(pertvar);
    user=pertnode->r->nuse(pertvar);
    if (usel && user) {
        pertnode->l->region_l(pertvar, L);
        pertnode->r->region_u(pertvar, U);
    }
    else if (usel)
    {
        pertnode->region(pertvar, L, U);
        pertnode->l->region_l(pertvar, L);
    }
    else {
        pertnode->region(pertvar, L, U);
        pertnode->r->region_u(pertvar, U);
    }
}

//-------------------
// suff statistics for all bottom nodes
/* Note: Worker class is from RcppParallel*/
struct AllSuffWorker: public Worker
{
    tree& x;
    xinfo& xi;
    dinfo& di;
    size_t nb;
    std::map<tree::tree_cp, size_t> bnmap;
    double* weight;
    
    // internal state
    double sdelta; size_t ysize;
    std::vector<double> sy;
    double n0;
    std::vector<double> n; //double n;
    
    
    std::vector<sinfo> sv_tmp;
    double *xx; //current x
    double *yy; //current y
    size_t ni; // index
    
    //constrcutor
    AllSuffWorker(tree& x, xinfo& xi, dinfo& di, std::map<tree::tree_cp, size_t> bnmap, size_t nb, double* weight): x(x), xi(xi),di(di),nb(nb),bnmap(bnmap),weight(weight){//, sy(di.p_y), n(di.p_y), sv_tmp(nb){
        sdelta=0;//n=0.0;
        ysize = di.p_y;
        sy.resize(di.p_y);
        n0=0.0; n.resize(di.p_y);
        sv_tmp.resize(nb);
        for (size_t k=0; k<nb; ++k) {
            sv_tmp[k].sy.resize(di.p_y);
            sv_tmp[k].n.resize(di.p_y);
            sv_tmp[k].ysize = di.p_y;
        }
    }
    
    AllSuffWorker(const AllSuffWorker& asw, Split):x(asw.x),xi(asw.xi),di(asw.di),nb(asw.nb),bnmap(asw.bnmap),weight(asw.weight){//, sy(asw.di.p_y), n(asw.di.p_y), sv_tmp(asw.nb){
        sdelta=0;//n=0.0;
        ysize = di.p_y;
        sy.resize(di.p_y);
        n0=0.0; n.resize(di.p_y);
        sv_tmp.resize(nb);
        for (size_t k=0; k<nb; ++k) {
            sv_tmp[k].sy.resize(di.p_y);
            sv_tmp[k].n.resize(di.p_y);
            sv_tmp[k].ysize = di.p_y;
        }
    }
    
    void operator()(std::size_t begin, std::size_t end){
        for (size_t i=begin; i<end; i++) {
            xx = di.x + i*di.p;
            yy = di.y + i*di.p_y;// y= di.y[i];
            /*std::vector<double> tmp_y;
            std::vector<double> tmp_w;
            for (size_t k=0; k<di.p_y; ++k) {
                tmp_y.push_back(yy[k]);
                tmp_w.push_back(weight[i*di.p_y+k]);
            }*/
            
            ni = bnmap[x.bot(xx, xi)]; // return node id of the leaf that xx falls in
            
            sv_tmp[ni].n0 += 1.0; sv_tmp[ni].sdelta += di.delta[i];
            for (size_t k=0; k<di.p_y; ++k) {
                sv_tmp[ni].n[k] += weight[i*di.p_y+k];
                sv_tmp[ni].sy[k] += weight[i*di.p_y+k]*yy[k];
            }
            //sv_tmp[ni].n += weight[i];
            //sv_tmp[ni].sy += weight[i]*y;
            //sv_tmp[ni].sy.resize(di.p_y); sv_tmp[ni].ysize = di.p_y;
            //std::transform(tmp_y.begin(), tmp_y.end(), tmp_w.begin(), tmp_y.begin(), std::multiplies<double>());
            //std::transform(sv_tmp[ni].sy.begin(), sv_tmp[ni].sy.end(), tmp_y.begin(), sv_tmp[ni].sy.begin(), std::plus<double>());
            //tmp_y.resize(0); tmp_w.resize(0);
        }
    }
    
    void join(const AllSuffWorker& asw){
        for (size_t i=0; i!=nb; i++) {
            sv_tmp[i].n0 += asw.sv_tmp[i].n0;
            sv_tmp[i].sdelta += asw.sv_tmp[i].sdelta;
            //sv_tmp[i].n += asw.sv_tmp[i].n;
            std::transform(sv_tmp[i].n.begin(), sv_tmp[i].n.end(), asw.sv_tmp[i].n.begin(), sv_tmp[i].n.begin(), std::plus<double>());
            //sv_tmp[i].sy += asw.sv_tmp[i].sy;
            std::transform(sv_tmp[i].sy.begin(), sv_tmp[i].sy.end(), asw.sv_tmp[i].sy.begin(), sv_tmp[i].sy.begin(), std::plus<double>());
        }
    }
};

void allsuff(tree& x, xinfo& xi, dinfo& di, double* weight, tree::npv& bnv, std::vector<sinfo>& sv)
{
    tree::tree_cp tbn; // pointer to tree bottom mode for the current obs
    size_t ni; // index in vector of current bottom nodes
    double *xx;
    double *yy;
    
    bnv.clear();
    
    x.getbots(bnv);
    
    typedef tree::npv::size_type bvsz;
    bvsz nb = bnv.size();
    sv.resize(nb);
    for (size_t k=0; k<nb; ++k) {
        sv[k].sy.resize(di.p_y);
        sv[k].n.resize(di.p_y);
        sv[k].ysize = di.p_y;
    }
    
    std::map<tree::tree_cp, size_t> bnmap;
    for (bvsz i=0; i!=bnv.size(); i++)
        bnmap[bnv[i]]=i;
    AllSuffWorker asw(x,xi,di,bnmap,nb,weight);
    parallelReduce(0, di.N, asw);
    
    for (size_t i=0; i!=nb; i++) {
        sv[i].n0 += asw.sv_tmp[i].n0;
        sv[i].sdelta += asw.sv_tmp[i].sdelta;
        //sv[i].n += asw.sv_tmp[i].n;
        std::transform(sv[i].n.begin(), sv[i].n.end(), asw.sv_tmp[i].n.begin(), sv[i].n.begin(), std::plus<double>());
        //sv[i].sy += asw.sv_tmp[i].sy;
        std::transform(sv[i].sy.begin(), sv[i].sy.end(), asw.sv_tmp[i].sy.begin(), sv[i].sy.begin(), std::plus<double>());
    }
}

// counts for all bottom nodes
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv)
{
    tree::tree_cp tbn;
    size_t ni;
    double *xx;
    double *yy;
    
    bnv.clear();
    x.getbots(bnv);
    
    typedef tree::npv::size_type bvsz;
    bvsz nb = bnv.size();
    
    std::vector<int> cts(bnv.size(),0);
    
    std::map<tree::tree_cp,size_t> bnmap;
    for (bvsz i=0; i!=bnv.size(); i++) bnmap[bnv[i]]=i;
    
    for (size_t i=0; i<di.N; i++) {
        xx = di.x + i*di.p; // vector for the ith subject
        yy = di.y + i*di.p_y;//y = di.y[i];
        
        tbn = x.bot(xx, xi); //find leaf for the ith subject;
        ni = bnmap[tbn]; // get its nid
        
        cts[ni] +=1; // count for this node, +1
    }
    return cts;
}
/*Note: cts is a vector of the number of obs within each leaf node of a tree*/

// update counts to reflect observation i?
void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, int sign)
{
    tree::tree_cp tbn;
    size_t ni;
    double *xx;
    double *yy;
    
    typedef tree::npv::size_type bvsz;
    bvsz nb = bnv.size();
    
    std::map<tree::tree_cp, size_t> bnmap;
    for (bvsz ii=1; ii!=bnv.size(); ii++) {
        bnmap[bnv[ii]]=ii;
    }
    
    xx = di.x + i*di.p;
    yy = di.y + i*di.p_y;//y = di.y[i];
    
    tbn = x.bot(xx, xi);
    ni = bnmap[tbn];
    
    cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, std::map<tree::tree_cp, size_t>& bnmap, int sign)
{
    tree::tree_cp tbn;
    size_t ni;
    double *xx;
    double *yy;
    // bnmap already given
    xx = di.x + i*di.p;
    yy = di.y + i*di.p_y;//y = di.y[i];
    
    tbn = x.bot(xx, xi);
    ni = bnmap[tbn];
    
    cts[ni] += sign;
    
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi, dinfo& di, std::map<tree::tree_cp, size_t>& bnmap, int sign, tree::tree_cp &tbn)
{
    // bnmap and tbn already exist;
    size_t ni;
    double *xx;
    double *yy;
    
    xx = di.x + i*di.p;
    yy = di.y + i*di.p_y;//y = di.y[i];
    
    tbn = x.bot(xx, xi);
    ni = bnmap[tbn];
    
    cts[ni] += sign;
}

// check min leaf size
bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo& di)
{
    bool good = true;
    tree::npv bnv;
    std::vector<int> cts;
    
    int m = 0;
    for (size_t tt=0; tt<t.size(); ++tt) {
        cts = counts(t[tt], xi, di, bnv);
        m = std::min(m, *std::min_element(cts.begin(), cts.end()));
        if (m<minct) {
            good = false;
            break;
        }
    }
    return good;
}

// ------------------------------
/*Note:skip all the MPI funcs for now*/
// ------------------------------

struct GetSuffBirthWorker: public Worker
{
    tree& x;
    tree::tree_cp nx;
    size_t v;
    size_t c;
    size_t p_mu;
    xinfo& xi;
    dinfo& di;
    double* phi;
    
    // internal state
    std::vector<double> l_n;
    std::vector<double> l_sy;
    double l_n0; double l_sdelta;
    std::vector<double> r_n;
    std::vector<double> r_sy;
    double r_n0; double r_sdelta;
    
    
    double *xx;
    double *yy;
    
    // Constructor
    
    GetSuffBirthWorker(tree &x,
                       tree::tree_cp nx,
                       size_t v,
                       size_t c,
                       size_t p_mu,
                       xinfo& xi,
                       dinfo& di,
                       double* phi):x(x),nx(nx),v(v),c(c),p_mu(p_mu),xi(xi),di(di),phi(phi){
        
        l_n.resize(di.p_y);//l_n=0.0;
        l_sy.resize(di.p_y);
        l_n0=0.0; l_sdelta=0.0;
        
        r_n.resize(di.p_y);//r_n=0.0;
        r_sy.resize(di.p_y);
        r_n0=0.0; r_sdelta=0.0;
    }
    // splitting constructor
    GetSuffBirthWorker(const GetSuffBirthWorker& gsw, Split):x(gsw.x),nx(gsw.nx),v(gsw.v),c(gsw.c),p_mu(gsw.p_mu),xi(gsw.xi),di(gsw.di),phi(gsw.phi){
        
        l_n.resize(di.p_y);//l_n=0.0;
        l_sy.resize(di.p_y);
        l_n0=0.0; l_sdelta=0.0;
        
        r_n.resize(di.p_y);//r_n=0.0;
        r_sy.resize(di.p_y);
        r_n0=0.0; r_sdelta=0.0;
    }
    
    // An operator() which performs the work
    void operator()(std::size_t begin, std::size_t end){
        for (size_t i=begin; i<end; i++) {
            xx = di.x + i*di.p;
            if (nx == x.bot(xx, xi)) {
                yy = di.y + i*di.p_y;//y = di.y[i];
             /*   std::vector<double> tmp_y;
                std::vector<double> tmp_w;
                for (size_t k=0; k<di.p_y; ++k) {
                    tmp_y.push_back(yy[k]);
                    tmp_w.push_back(phi[i*di.p_y+k]);
                }*/
                if (xx[v] < xi[v][c]) {
                    l_n0 += 1; l_sdelta += di.delta[i];
                    for (size_t k=0; k<di.p_y; ++k) {
                        l_n[k] += phi[i*di.p_y+k];
                        l_sy[k] += phi[i*di.p_y+k]*yy[k];
                    }
                    //l_n += phi[i]; //weights
                    //std::transform(l_n.begin(), l_n.end(), tmp_w.begin(), l_n.begin(), std::plus<double>());
                    //l_sy += phi[i]*y;
                    //std::transform(tmp_y.begin(), tmp_y.end(), tmp_w.begin(), tmp_y.begin(), std::multiplies<double>());
                    //std::transform(l_sy.begin(), l_sy.end(), tmp_y.begin(), l_sy.begin(), std::plus<double>()); //[tmp_w](double i, double j) {return i+tmp_w*j; });
                    //tmp_y.clear(); tmp_w.clear();
                } else {
                    r_n0 += 1; r_sdelta += di.delta[i];
                    for (size_t k=0; k<di.p_y; ++k) {
                        r_n[k] += phi[i*di.p_y+k];
                        r_sy[k] += phi[i*di.p_y+k]*yy[k];
                    }
                    //r_n += phi[i];
                    //std::transform(r_n.begin(), r_n.end(), tmp_w.begin(), r_n.begin(), std::plus<double>());
                    //r_sy += phi[i]*y;
                    //std::transform(tmp_y.begin(), tmp_y.end(), tmp_w.begin(), tmp_y.begin(), std::multiplies<double>());
                    //std::transform(r_sy.begin(), r_sy.end(), tmp_y.begin(), r_sy.begin(), std::plus<double>()); //[tmp_w](double i, double j) {return i+tmp_w*j; });
                    //tmp_y.clear(); tmp_w.clear();
                }
            }
        }
    }
    
    void join(const GetSuffBirthWorker& gsw){
        //l_n += gsw.l_n;
        std::transform(l_n.begin(), l_n.end(), gsw.l_n.begin(), l_n.begin(), std::plus<double>());
        //l_sy += gsw.l_sy;
        std::transform(l_sy.begin(), l_sy.end(), gsw.l_sy.begin(), l_sy.begin(), std::plus<double>());
        l_n0 += gsw.l_n0; l_sdelta += gsw.l_sdelta;
        
        //r_n += gsw.r_n;
        std::transform(r_n.begin(), r_n.end(), gsw.r_n.begin(), r_n.begin(), std::plus<double>());
        //r_sy += gsw.r_sy;
        std::transform(r_sy.begin(), r_sy.end(), gsw.r_sy.begin(), r_sy.begin(), std::plus<double>());
        r_n0 += gsw.r_n0; r_sdelta += gsw.r_sdelta;
    }
};

// get sufficient stat for children (v,c) of node nx in tree x
void getsuffBirth(tree& x, tree::tree_cp nx, size_t v, size_t c, size_t p_mu, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr)
{
    GetSuffBirthWorker gsw(x,nx,v,c,p_mu,xi,di,phi);
    
    parallelReduce(0,di.N,gsw);
    
    if (gsw.p_mu != di.p_y) {
        Rcpp::Rcout << "error: p_mu(" << gsw.p_mu <<") and di.p_y(" << di.p_y <<")  unmatched\n";
        return;
    }
    
    sl.n = gsw.l_n;
    sl.sy = gsw.l_sy;
    sl.n0 = gsw.l_n0;
    sl.ysize = gsw.p_mu;
    sl.sdelta = gsw.l_sdelta;
    
    sr.n = gsw.r_n;
    sr.sy = gsw.r_sy;
    sr.n0 = gsw.r_n0;
    sr.ysize = gsw.p_mu;
    sr.sdelta = gsw.r_sdelta;
}

struct GetSuffDeathWorker: public Worker
{
    tree& x;
    tree::tree_cp nl;
    tree::tree_cp nr;
    xinfo& xi;
    dinfo& di;
    double* phi;
    
    // internal state
    std::vector<double> l_n;
    std::vector<double> l_sy;
    double l_n0; double l_sdelta;
    
    std::vector<double> r_n;
    std::vector<double> r_sy;
    double r_n0; double r_sdelta;
    
    double *xx;
    double *yy;
    
    //constructor
    
    GetSuffDeathWorker(tree& x,
                       tree::tree_cp nl,
                       tree::tree_cp nr,
                       xinfo& xi,
                       dinfo& di,
                       double* phi):x(x),nl(nl),nr(nr),xi(xi),di(di),phi(phi){
        
        l_n.resize(di.p_y);//l_n = 0.0;
        l_sy.resize(di.p_y);
        l_n0 = 0.0; l_sdelta = 0.0;
        r_n.resize(di.p_y);//r_n = 0.0;
        r_sy.resize(di.p_y);
        r_n0 = 0.0; r_sdelta = 0.0;
    }
    
    //splitting constructor
    GetSuffDeathWorker(const GetSuffDeathWorker& gsw, Split):x(gsw.x),nl(gsw.nl),nr(gsw.nr),xi(gsw.xi),di(gsw.di),phi(gsw.phi){
        
        l_n.resize(di.p_y);//l_n = 0.0;
        l_sy.resize(di.p_y);
        l_n0 = 0.0; l_sdelta = 0.0;
        r_n.resize(di.p_y);//r_n = 0.0;
        r_sy.resize(di.p_y);
        r_n0 = 0.0; r_sdelta = 0.0;
    }
    
    void operator()(std::size_t begin, std::size_t end){
        for (size_t i=begin; i<end; i++) {
            xx = di.x + i*di.p;
            tree::tree_cp bn = x.bot(xx, xi);
            yy = di.y + i*di.p_y;//y = di.y[i];
            /*std::vector<double> tmp_y;
            std::vector<double> tmp_w;
            for (size_t k=0; k<di.p_y; ++k) {
                tmp_y.push_back(yy[i]);
                tmp_w.push_back(phi[i*di.p_y+k]);
            }*/
            
            if (bn==nr) {
                r_n0 += 1; r_sdelta += di.delta[i];
                for (size_t k=0; k<di.p_y; ++k) {
                    r_n[k] += phi[i*di.p_y+k];
                    r_sy[k] += phi[i*di.p_y+k]*yy[k];
                }
                //r_n += phi[i];
                //std::transform(r_n.begin(), r_n.end(), tmp_w.begin(), r_n.begin(), std::plus<double>());
                //r_sy += phi[i]*y;
                //std::transform(tmp_y.begin(), tmp_y.end(), tmp_w.begin(), tmp_y.begin(), std::multiplies<double>());
                //std::transform(r_sy.begin(), r_sy.end(), tmp_y.begin(), r_sy.begin(), std::plus<double>());//[tmp_w](double i, double j) {return i+tmp_w*j; });
                //tmp_y.clear(); tmp_w.clear();
            }
            if (bn == nl) {
                l_n0 += 1; l_sdelta += di.delta[i];
                for (size_t k=0; k<di.p_y; ++k) {
                    l_n[k] += phi[i*di.p_y+k];
                    l_sy[k] += phi[i*di.p_y+k]*yy[k];
                }
                //l_n += phi[i];
                //std::transform(l_n.begin(), l_n.end(), tmp_w.begin(), l_n.begin(), std::plus<double>());
                //l_sy += phi[i]*y;
                //std::transform(tmp_y.begin(), tmp_y.end(), tmp_w.begin(), tmp_y.begin(), std::multiplies<double>());
                //std::transform(l_sy.begin(), l_sy.end(), tmp_y.begin(), l_sy.begin(), std::plus<double>());//[tmp_w](double i, double j) {return i+tmp_w*j; });
                //tmp_y.clear(); tmp_w.clear();
            }
        }
    }
    
    void join(const GetSuffDeathWorker& gsw){
        //l_n += gsw.l_n;
        std::transform(l_n.begin(), l_n.end(), gsw.l_n.begin(), l_n.begin(), std::plus<double>());
        //l_sy += gsw.l_sy;
        std::transform(l_sy.begin(), l_sy.end(), gsw.l_sy.begin(), l_sy.begin(), std::plus<double>());
        l_n0 += gsw.l_n0; l_sdelta += gsw.l_sdelta;
        
        //r_n += gsw.r_n;
        std::transform(r_n.begin(), r_n.end(), gsw.r_n.begin(), r_n.begin(), std::plus<double>());
        //r_sy += gsw.r_sy;
        std::transform(r_sy.begin(), r_sy.end(), gsw.r_sy.begin(), r_sy.begin(), std::plus<double>());
        r_n0 += gsw.r_n0; r_sdelta += gsw.r_sdelta;
    }
    
};

// get sufficient stat for pair of bottom children nl, nr, in tree x
void getsuffDeath(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, double* phi, sinfo& sl, sinfo& sr)
{
    GetSuffDeathWorker gsw(x,nl,nr,xi,di,phi);
    parallelReduce(0, di.N, gsw);
    
    sl.n = gsw.l_n;
    sl.sy = gsw.l_sy;
    sl.n0 = gsw.l_n0;
    sl.ysize = gsw.di.p_y;
    sl.sdelta = gsw.l_sdelta;
    
    sr.n = gsw.r_n;
    sr.sy = gsw.r_sy;
    sr.n0 = gsw.r_n0;
    sr.ysize = gsw.di.p_y;
    sr.sdelta = gsw.r_sdelta;
}

// ------------------------------
/*Note:skip all the MPI funcs for now*/
// ------------------------------
// log integrated likelihood, here mu_j of the node is integrated out.
// the log integrated like is calculated for each single bottom node
double loglike(double n, double sy, double sigma, double tau)
{
    double d = 1/(tau*tau) + n; // n=\sum phi_i or n0/sigma^2? for heterogeneous?
    
    double out = -log(tau) - 0.5*log(d);
    out += 0.5*sy*sy/d;
    return out;
}

double loglikelg(double sdelta, double sy, double lg_alpha, double lg_beta)
{
    double new_al = sdelta + lg_alpha; // note that sdelta here is either \sum\delta_i or  \sum\delta_iA_i
    double new_bt = lg_beta + sy;      // and sy here is \sum exp(res) \sum:k:i in R_k \lambda_k\delta_k
    double out =  lg_alpha*log(lg_beta) - std::lgamma(lg_alpha) + std::lgamma(new_al) - new_al*log(new_bt); //lg_alpha*log(lg_beta) - log(std::tgamma(lg_alpha)) +
    return out;
}

double loglike_mvn(double n, arma::vec sy, arma::mat covsigma, arma::mat covprior)
{
    double det1 = arma::det(covprior);
    arma::mat postSig = (n*covsigma.i() + covprior.i()).i();
    double det2 = arma::det(postSig);
    double out = 0.5*log(det2)-0.5*log(det1);
    //out += 0.5 * (sy.t() * covsigma.i() * postSig * covsigma.i() * sy).eval()(0,0);
    arma::vec myvec = sy.t() * covsigma.i() * postSig * covsigma.i() * sy;
    out += 0.5 * (arma::conv_to < std::vector<double> >::from(myvec))[0];
    return out;
}

/*
struct FitWorker: public Worker
{
    tree& t;
    xinfo& xi;
    dinfo& di;
    
    // internal
    double *xx;
    tree::tree_cp bn;
    std::vector<double> &fv; //node means
    
    //constructor
    FitWorker(tree& t,
              xinfo& xi,
              dinfo& di,
              std::vector<double>& fv):t(t),xi(xi),di(di),fv(fv){ }
    
    // No split constructor?
    
    void operator()(std::size_t begin, std::size_t end){
        for (size_t i=begin; i<end; i++) {
            xx = di.x + i*di.p;
            bn = t.bot(xx, xi);
            fv[i] = bn->getmu();
        }
    }
    
    void join(const FitWorker& fir){ }
};

void fit(tree& t, xinfo& xi, dinfo& di, std::vector<double>& fv)
{
    fv.resize(di.N);
    
    FitWorker fir(t,xi,di,fv);
    parallelReduce(0, di.N, fir);
}
*/
// without parallel work?
void fit(tree& t, xinfo& xi, dinfo& di, double* fv)
{
    double* xx;
    tree::tree_cp bn;
    for (size_t i=0; i<di.N; i++) {
        xx = di.x + i*di.p;
        bn = t.bot(xx, xi);
        for (size_t j=0; j<di.p_y; j++) {
            fv[i*di.p_y+j] = bn->getmu()[j];//fv[i] = bn->getmu();
        }
    }
}

void predict(std::vector<tree>& tv, xinfo& xi, dinfo& di, double* fp)
{
    size_t ntemp = di.N;
    size_t py = di.p_y;
    double* fptemp = new double[ntemp*py];
    
    for (size_t k=0; k<ntemp*py; k++) fp[k]=0.0;
    for (size_t j=0; j<tv.size(); j++) {
        fit(tv[j], xi, di, fptemp);
        for (size_t k=0; k<ntemp*py; k++) {
            fp[k] += fptemp[k];
        }
    }
    delete [] fptemp;
}

void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv)
{
    double *xx;
    tree::tree_cp bn;
    pv.resize(di.N);
    for (size_t i=0; i<di.N; i++) {
        xx = di.x + i*di.p;
        bn = t.bot(xx, xi);
        pv[i] = bn->nid();
    }
}

// draw all bottom nodes mu
void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double* weight, RNG& gen, arma::mat Covsig, arma::mat Covprior)
{
    tree::npv bnv;
    std::vector<sinfo> sv;
    
    allsuff(t, xi, di, weight, bnv, sv);
    std::vector<double> fcmean(di.p_y);
    //fcmean.resize(di.p_y);
    std::vector<double> eps(di.p_y);
    //eps.resize(di.p_y);
    std::vector<double> fcvar(di.p_y);
    //fcvar.resize(di.p_y);
    // bottom nodes stored in Expecting a single value: [extent=0].bnv
    // suff stat stores in sv, get from data
    // Do NOTE: sv[i].n is actually n/sigma^2 since weight = 1/sigma^2; and sv[i].y is thus \sum_y/sigma^2
    for (tree::npv::size_type i=0; i!=bnv.size(); i++) {
        //double fcvar = 1.0/(1.0/(pi.tau*pi.tau)+sv[i].n);
        /*
        for (size_t k=0; k<di.p_y; ++k) {
            fcvar[k] = 1.0/(1.0/(pi.tau[k]*pi.tau[k])+sv[i].n[k]);
        }*/
        arma::mat postSig = (sv[i].n0 * Covsig.i() + Covprior.i()).i();
        arma::vec svsy(di.p_y, arma::fill::zeros);
        for (size_t k=0; k<di.p_y; ++k) {
            svsy(k) = sv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
        }
        arma::vec arma_fcmean = Covsig.i() * postSig * svsy;
        arma::mat sample = gen.mvnorm(arma_fcmean, postSig);
        //double fcmean = sv[i].sy*fcvar;
        //std::transform(sv[i].sy.begin(), sv[i].sy.end(), fcmean.begin(), [fcvar](double i) { return i*fcvar; });
        /*
        std::transform(sv[i].sy.begin(), sv[i].sy.end(), fcvar.begin(), fcmean.begin(), std::multiplies<double>());
        for (size_t k=0; k<di.p_y; ++k) {
            eps[k] = gen.normal()*sqrt(fcvar[k]);
        }
        std::transform(fcmean.begin(), fcmean.end(), eps.begin(), fcmean.begin(), std::plus<double>());
        */
        for (size_t k=0; k<di.p_y; ++k) {
            fcmean[k] = sample(k);
        }
        /*Rcout << fcvar << endl;
        Rcout << sv[i].sy << endl;
        Rcout << eps << endl;
        Rcout << fcmean << endl;*/
        bnv[i]->setmu(fcmean);//bnv[i]->setmu(fcmean + gen.normal()*sqrt(fcvar));
        
        if (bnv[i]->getmu() != bnv[i]->getmu()) {
            for (int j=0; j<di.N; ++j) Rcout << *(di.x + j*di.p) << " "; // print covariate matrix
            Rcout << endl << "fcvar" << fcvar << "svi[n]" << sv[i].n << "i" << i;
            Rcout << endl << t;
            Rcpp::stop("drmu failed");
        }
    }
}

void prxi(xinfo& xi)
{
    Rcout << "xinfo:\n";
    for (size_t v=0; v!=xi.size(); v++) {
        Rcout << "v: " << v << endl;
        for (size_t j=0; j!=xi[v].size(); j++) {
            Rcout << "j,xi[v][j]: " << j << "," << xi[v][j] << endl;
        }
    }
    Rcout << "\n\n";
}

void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
    double xinc;
    
    std::vector<double> minx(p, INFINITY);
    std::vector<double> maxx(p, -INFINITY);
    double xx;
    for (size_t i=0; i<p; i++) {
        for (size_t j=0; j<n; j++) {
            xx = *(x+p*j+i);
            if (xx < minx[i]) minx[i]=xx;
            if (xx > maxx[i]) maxx[i]=xx;
        }
    }
    
    // make grid of nc cutpoints between min and max for each x
    xi.resize(p);
    for (size_t i=0; i<p; i++) {
        xinc = (maxx[i] - minx[i])/(nc+1.0);
        xi[i].resize(nc);
        for (size_t j=0; j<nc; j++) xi[i][j] = minx[i] + (j+1)*xinc;
    }
}

// get min/max for p predictors needed to make cutpoints
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx)
{
    double xx;
    
    for (size_t i=0; i<p; i++) {
        for (size_t j=0; j<n; j++) {
            xx = *(x+p*j+i);
            if (xx < minx[i]) minx[i]=xx;
            if (xx > maxx[i]) maxx[i]=xx;
        }
    }
    
}

// make xinfo given min/max
void makexinfominmax(size_t p, xinfo&xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx)
{
    double xinc;
    xi.resize(p);
    for (size_t i=0; i<p; i++) {
        xinc = (maxx[i] - minx[i])/(nc+1.0);
        xi[i].resize(nc);
        for (size_t j=0; j<nc; j++) {
            xi[i][j] = minx[i] + (j+1)*xinc;
        }
    }
}

void updateLabels(int* labels, double* mixprop, double* locations, double* resid, double sigma, size_t nobs, size_t nmix, RNG& gen)
{
    // use random uniform generator to sample from multinomial
    double ptot, pcum, u;
    int count;
    for (size_t k=0; k<nobs; k++) {
        ptot = 0.0;
        count = 1;
        pcum = 0.0;
        for (size_t h=0; h<nmix; h++){
            ptot += mixprop[h]*(1/sqrt(2*PI*sigma*sigma))*exp(-0.5*(resid[k]-locations[h])*(resid[k]-locations[h])/(sigma*sigma));//R::dnorm(resid[k], locations[h], sigma, 0);
        }
        //u = ptot*R::runif(0.0, 1.0);
        u = ptot*gen.uniform();
        for (size_t h=0; h<nmix; h++) {
            pcum += mixprop[h]*(1/sqrt(2*PI*sigma*sigma))*exp(-0.5*(resid[k]-locations[h])*(resid[k]-locations[h])/(sigma*sigma));//R::dnorm(resid[k], locations[h], sigma, 0);
            if (u < pcum){
                break;
            }
            count += 1;
        }
        labels[k] = count; // Note: so the resulted count/labels will be 1 ~ nmix
    }
}

void updateMixprp(double* mixprop, double* mass, int* labels, int* mixcnts, size_t nmix, double psi1, double psi2, RNG& gen)
{
    int ncum;
    double shape1, shape2, gam_shape, gam_scale;
    double Vold, Vnew, log_mixprp, Vcum, tmp;
    
    ncum = std::accumulate(mixcnts+1, mixcnts+nmix, 0);
    shape1 = mixcnts[0] + 1.0;
    shape2 = mass[0] + ncum;
    Vold = gen.beta(shape1, shape2);
    //Vold = R::rbeta(shape1, shape2);
    log_mixprp = log(Vold);
    Vcum = log(1-Vold); // for updating mass
    mixprop[0] = Vold;
    for (size_t h=1; h<nmix-1; h++) {
        ncum = std::accumulate(mixcnts+h+1, mixcnts+nmix, 0);
        shape1 = mixcnts[h] + 1.0;
        shape2 = mass[0] + ncum;
        Vnew = gen.beta(shape1, shape2);
        //Vnew = R::rbeta(shape1, shape2);
        tmp = (Vnew/Vold)*(1-Vold);
        log_mixprp += log(tmp); // the log v_h
        
        Vcum += log1p(-Vnew); // for updating mass
        Vold = Vnew;
        mixprop[h] = exp(log_mixprp);
    }
    mixprop[nmix-1] = 1.0 - std::accumulate(mixprop, mixprop + nmix - 1, 0.0);
   
    gam_shape = psi1 + nmix - 1;
    gam_scale = 1/(psi2 - Vcum);

    //mass[0] = gen.gamma(psi1 + nmix - 1, 1/(psi2 - Vcum));
    mass[0] = R::rgamma(gam_shape, gam_scale);
   //Rprintf("Gamma draw': %f \n", Vcum);
}

void updateLocations(double* locations, double* mixprop, double* resid, int* labels, int* mixcnts, double sigma, double prior_sigsq, size_t nobs, size_t nmix, RNG& gen)
{
    double sigsq, compo_sum, compo_include, post_prec, post_mean, post_sd, muG;
    
    sigsq = sigma*sigma;
    for (size_t h=0; h<nmix; h++) {
        compo_sum = 0.0;
        for (size_t i=0; i<nobs; i++) {
            if (labels[i] == h+1) { // recall that the labels are 1,...,nmix
                compo_include = 1.0;
            } else {
                compo_include = 0.0;
            }
            //compo_include = (labels[i] == h+1) ? 1.0 : 0.0;
            compo_sum += compo_include*resid[i];
        }
        post_prec = 1/(prior_sigsq*mixcnts[h] + sigsq);
        post_mean = prior_sigsq*post_prec*compo_sum;
        post_sd = sqrt(sigsq*prior_sigsq*post_prec);
        locations[h] = gen.normal(post_mean, post_sd);
        
        //locations[h] = R::rnorm(post_mean, post_sd);
    }
    // normalization of the locations
    muG = 0.0;
    for (size_t h=0; h<nmix; h++) {
        muG += mixprop[h]*locations[h];
    }
    
    for (size_t h=0; h<nmix; h++) {
        locations[h] -= muG;
    }
}

void updateIndivLocations (double* indiv_locations, double* locations, int* labels, size_t nobs, size_t nmix)
{
    double loc = 0.0;
    for (size_t k=0; k<nobs; k++) {
        for (size_t h=0; h<nmix; h++) {
            if (labels[k] == h+1) {
                loc = locations[h];
                break;
            }
        }
        indiv_locations[k] = loc;
    }
}

double logPriT(tree::tree_p x, xinfo& xi, pinfo& pi)
{
    double p_grow = pgrow(x, xi, pi);
    double retval = 0.0;
    size_t v;
    std::vector<size_t> goodvars;
    int L,U;
    if (x->ntype() == 'b') {
        retval = log(1.0-p_grow);
    } else {
        retval = log(p_grow);
        getgoodvars(x, xi, goodvars);
        retval += log(1.0/(double)goodvars.size());
        v = x->getv();
        L=0;U=xi[v].size()-1;
        x->region(v, &L, &U);
        retval -= log((double)(U-L+1));
        retval += logPriT(x->getl(), xi, pi) + logPriT(x->getr(), xi, pi);
    }
    return retval;
}

bool CheckRule(tree::tree_p n, xinfo& xi, size_t var)
{
    int L,U;
    bool goodrule = false;
    L=0; U=xi[var].size()-1;
    n->region(var, &L, &U);
    if (!(n->getl())) {
        goodrule = true;
    } else {
        size_t v = n->getv();
        if (v == var) {
            size_t c = n->getc();
            if ((c>=L) && (c<=U)) {
                goodrule = (CheckRule(n->getl(), xi, var) && CheckRule(n->getl(), xi, var));
            }
        } else {
            goodrule = (CheckRule(n->getl(), xi, var) && CheckRule(n->getl(), xi, var));
        }
    }
    return goodrule;
}

void drmu_withlg(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double* weight, RNG& gen, arma::mat Covsig, arma::mat Covprior, std::vector<double>& storevec1, std::vector<double>& storevec2, std::vector<double>& storemu)
{
    tree::npv bnv;
    std::vector<sinfo> sv;
    
    allsuff(t, xi, di, weight, bnv, sv);
    std::vector<double> fcmean;
    fcmean.resize(di.p_y);
    std::vector<double> eps;
    eps.resize(di.p_y-1);
    std::vector<double> fcvar;
    fcvar.resize(di.p_y);
    // bottom nodes stored in Expecting a single value: [extent=0].bnv
    // suff stat stores in sv, get from data
    // Do NOTE: sv[i].n is actually n/sigma^2 since weight = 1/sigma^2; and sv[i].y is thus \sum_y/sigma^2
    for (tree::npv::size_type i=0; i!=bnv.size(); i++) {
        //double fcvar = 1.0/(1.0/(pi.tau*pi.tau)+sv[i].n);
        if (di.p_y > 1) {
            arma::mat postSig = (sv[i].n0 * Covsig.i() + Covprior.i()).i();
            arma::vec svsy(di.p_y-1, arma::fill::zeros);
            for (size_t k=0; k<di.p_y-1; ++k) {
                svsy(k) = sv[i].sy[k]*pi.sigma[k]*pi.sigma[k];
            }
            arma::vec arma_fcmean = Covsig.i() * postSig * svsy;
            arma::mat sample = gen.mvnorm(arma_fcmean, postSig);
            
            for (size_t k=0; k<di.p_y-1; ++k) {
                fcmean[k] = sample(k);
            }
            /*
            for (size_t k=0; k<di.p_y-1; ++k) {
                fcvar[k] = 1.0/(1.0/(pi.tau[k]*pi.tau[k])+sv[i].n[k]);
                fcmean[k] = sv[i].sy[k]*fcvar[k];
                fcmean[k] += gen.normal()*sqrt(fcvar[k]);
            }*/
        }
        double fc_alpha = pi.lg_alpha + sv[i].sdelta;
        double fc_beta = pi.lg_beta + sv[i].sy[di.p_y-1];
        storevec1.push_back(sv[i].sdelta);
        storevec2.push_back(sv[i].sy[di.p_y-1]);
        //Rcout << "fc_alpha: " << fc_alpha << "; fc_beta " << fc_beta << endl;
        fcmean[di.p_y-1] = gen.loggamma(fc_alpha, 1/fc_beta);
        
        bnv[i]->setmu(fcmean);
        storemu.push_back(fcmean[di.p_y-1]);
        
        if (bnv[i]->getmu() != bnv[i]->getmu()) {
            for (int j=0; j<di.N; ++j) Rcout << *(di.x + j*di.p) << " "; // print covariate matrix
            Rcout << endl << "fcvar" << fcvar << "svi[n]" << sv[i].n << "i" << i;
            //Rcout << endl << "fcvar" << fcmean << "svi[n]" << sv[i].n << "i" << i;
            Rcout << endl << t;
            Rcpp::stop("drmu failed");
        }
    }
}

//risk set
void riskset(std::vector<double> &yori, std::vector<double> &h0, std::vector<double> &hintv, std::vector<double> &whichinv, double* weight)
{
    size_t n = yori.size();
    size_t nord = hintv.size();
    //std::vector<double> tmp(n,0.0);
    double tmp = 0.0;
    for (size_t i=0; i<n; i++) {
        tmp = 0.0;
        tmp += h0[whichinv[i]]*(yori[i] - hintv[whichinv[i]]);
        if (whichinv[i]>0) {
            for (size_t k=0; k<whichinv[i]; ++k) {
                tmp += h0[k]*(hintv[k+1] - hintv[k]);
            }
        }
        weight[i] = tmp;
    }
}


/*
void riskset(std::vector<double> &yori, std::vector<double> &h, std::vector<double> &yord, std::vector<double> &delta, double* weight)
{
    size_t n = yori.size();
    size_t nord = yord.size();
    //std::vector<double> tmp(n,0.0);
    double tmp = 0.0;
    for (size_t i=0; i<n; ++i) {
        tmp = 0.0;
        for (size_t k=0; k<nord; ++k) {
            if (yori[i] >= yord[k]) {
                // find the yori[k] in yord
                tmp += h[k];
            }
        }
        weight[i] = tmp;
    }
}*/

size_t sample_class(const std::vector<double> &probs, RNG& gen)
{
    double u = gen.uniform();
    size_t K = probs.size();
    double foo = 0.0;
    
    for (size_t k=0; k<K; ++k) {
        foo += probs[k];
        if (u < foo) {
            return k;
        }
    }
    return K-1;
}

double log_sum_exp(const std::vector<double> &x)
{
    double M = *std::max_element(std::begin(x), std::end(x));
    double tmp = 0;
    for (size_t k=0; k<x.size(); ++k) {
        tmp += exp(x[k] - M);
    }
    return M + log(tmp);
}


void UpdateS(dinfo& di, pinfo& pi, RNG& gen, std::vector<size_t> &ivcnt, std::vector<double> &S)
{
    size_t size_S = S.size();
    if (size_S!=di.p) {
        Rcout << "error in size: p_x!=S.size()\n";
        return;
    }
    
    if (size_S!=ivcnt.size()) {
        Rcout << "error in size: S.size!=ivcnt.size\n";
        return;
    }
    
    std::vector<double> shape_up(size_S, 0.);
    std::vector<double> tmp_logS(size_S, 0.);
    for (size_t k=0; k<size_S; ++k) {
        shape_up[k] = pi.a_drch/((double)size_S) + (double)ivcnt[k];
        tmp_logS[k] = gen.rlgam(shape_up[k]);
    }
    
    double tmp = log_sum_exp(tmp_logS);
    
    for (size_t i=0; i<size_S; ++i) {
        S[i] = exp(tmp_logS[i] - tmp);
    }
}


bool isleaf(tree::tree_p t) {
    bool isleaf = false;
    if (t->ntype()=='b' || t->treesize() == 1) {
        isleaf = true;
    }
    return isleaf;
}

void AddTreeCounts(ProbHypers& hypers, tree::tree_p x)
{
    if (!isleaf(x)) {
        hypers.counts[x->getv()] = hypers.counts[x->getv()] + 1;
        AddTreeCounts(hypers, x->getl());
        AddTreeCounts(hypers, x->getr());
    }
}

void SubtractTreeCounts(ProbHypers& hypers, tree::tree_p x)
{
    if (!isleaf(x)) {
        hypers.counts[x->getv()] = hypers.counts[x->getv()] - 1;
        SubtractTreeCounts(hypers, x->getl());
        SubtractTreeCounts(hypers, x->getr());
    }
}

double cutpoint_likelihood(tree::tree_p x, xinfo& xi)
{
    if (isleaf(x)) return 1.;
    
    int L,U;
    size_t v = x->getv();
    L=0; U=xi[v].size()-1;
    getpertLU(x, v, xi, &L, &U);
    double out = 1./(xi[v][U] - xi[v][L]);
    out = out * cutpoint_likelihood(x->getl(), xi);
    out = out * cutpoint_likelihood(x->getr(), xi);
    
    return out;
    
}

void GenBelow(tree::tree_p x, pinfo& pi, dinfo& di, xinfo& xi, double* phi, ProbHypers& hypers, RNG& gen, bool vs)
{
    double pgrow = pi.alpha/pow(1.0+x->depth(), pi.beta);
    double u = gen.uniform();
    if (u < pgrow) {
        if (isleaf(x)) {
            tree::tree_p l = new tree;
            l->p_mu = di.p_y;
            l->mu = std::vector<double>(l->p_mu, 0.);
            tree::tree_p r = new tree;
            r->p_mu = di.p_y;
            r->mu = std::vector<double>(r->p_mu, 0.);
            x->l = l;
            x->r = r;
            l->p = x;
            r->p = x;
            
            std::vector<size_t> goodvars;
            getgoodvars(x, xi, goodvars); // get variable this node can split on
            
            size_t vi = floor(gen.uniform()*goodvars.size());
            size_t v = goodvars[vi];
            
            if (vs) {
                v = hypers.SampleVar(gen);
                //hypers.counts[v]+=1;
            }
            hypers.counts[v]+=1;
            int L,U;
            L=0; U=xi[v].size()-1;
            x->region(v, &L, &U);
            size_t c = L + floor(gen.uniform()*(U-L+1));
            x->v = v;
            x->c = c;
            sinfo sl,sr;
            getsuffBirth(*x, x, v, c, di.p_y, xi, di, phi, sl, sr);
            if ((sl.n0>4) && (sr.n0>4)) {
                GenBelow(l, pi, di, xi, phi, hypers, gen, vs);
                GenBelow(r, pi, di, xi, phi, hypers, gen, vs);
            }
            
        }
    }
}


double BSinvTrigamma(double lower, double upper, double x) {
    if (upper >= lower) {
        double mid = lower + (upper - lower)/2;
        if (R::trigamma(mid) > x) {
            if ((R::trigamma(mid) - x) < 0.001) {
                return mid;
            } else {
                return BSinvTrigamma(mid, upper, x);
            }
        } else if ((x - R::trigamma(mid)) < 0.001) {
            return mid;
        } else {return BSinvTrigamma(lower, mid, x);}
    } else Rcpp::stop("Find invTrigamma failed");
}


// Function to compute the B-spline basis function value
double bspline(double x, int i, int degree, const std::vector<double>& knots) {
    if (degree == 0) {
        if ((x >= knots[i]) && (x < knots[i+1]))
            return 1.0;
        return 0.0;
    }

    double denom1 = knots[i + degree] - knots[i];
    double denom2 = knots[i + degree + 1] - knots[i + 1];

    double coeff1 = 0.0;
    double coeff2 = 0.0;

    if (denom1 > 0.0) {
        coeff1 = (x - knots[i]) / denom1 * bspline(x, i, degree - 1, knots);
    }

    if (denom2 > 0.0) {
        coeff2 = (knots[i + degree + 1] - x) / denom2 * bspline(x, i + 1, degree - 1, knots);
    }

    return coeff1 + coeff2;
}

// Function to get the Bspline fit for the hazard
double GetHZD(double x, int degree, const std::vector<double>& knots, double* gamma) {
    double basis = 0.;
    double hzd = 0.;
    for (size_t j=0; j < (knots.size() - degree - 1); ++j) {
        basis = std::max(0., bspline(x, j, degree, knots));
        hzd += gamma[j]*basis;
    }
    return exp(hzd);
}

double GetCumhzd(double x, int degree, size_t ngrids, const std::vector<double>& knots, double* gamma) {
    double h = x/ngrids;
    
    double sum = GetHZD(0., degree, knots, gamma) + GetHZD(x, degree, knots, gamma);
    
    for (size_t m=1; m<ngrids; m++) {
        sum += 2*GetHZD(m*h, degree, knots, gamma);
    }
    
    return (h/2)*sum;
}

double gammaFC(double rho, double eps, int degree, size_t ngrids, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma) {
    double sum_prior = 0.;
    double sum_lhd = 0.;
    double sum_iden = 0.;
    size_t n = event.size();
    std::vector<double> hzd(n, 0.);
    std::vector<double> cumhzd(n, 0.);
    
    for (size_t k=0; k < (knots.size() - degree - 1); ++k) {
        sum_prior += eps*gamma[k]*gamma[k];
        if ((k > 0) && (k < knots.size() - degree - 2)) {
            sum_prior += pow(gamma[k-1] -2*gamma[k] + gamma[k+1], 2);
        }
    }
    
    
    for (size_t i=0; i < n; ++i) {
        hzd[i] = GetHZD(yobs[i], degree, knots, gamma);
        cumhzd[i] = GetCumhzd(yobs[i], degree, ngrids, knots, gamma);
        sum_lhd += event[i]*log(hzd[i]) - cumhzd[i]*exp(allfit[i]);
        //for (size_t j=0; j < (knots.size() - degree - 1); ++j) {
        //    sum_iden += gamma[j]*std::max(0., bspline(yobs[i], j, degree, knots));
        //}
    }
    
    
    if (sum_iden == 0) {
        return (sum_lhd - 0.5*rho*sum_prior);
    } else {
        return (sum_lhd - INFINITY);
    }
    
    //return (sum_lhd - 0.5*rho*sum_prior);
    
}


void gammaMH(double rho, double eps, int degree, size_t ngrids, double stepsize, double* MHcounts_gamma, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma, RNG& gen) {
    
    size_t L = (knots.size() - degree - 1);
    double* candidate = new double[L];
    for (size_t k=0; k<L; ++k) {
        candidate[k] = gamma[k];
    }
    
    // univariate Metropolis steps
    for (size_t k=0; k<L; ++k) {
        candidate[k] = gamma[k] + stepsize*gen.normal();
        
        double loglike_old = gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, gamma);
        double loglike_new = gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, candidate);
        double alpha = std::min(1.0, exp(loglike_new - loglike_old));
        
        if (gen.uniform() < alpha) {
            gamma[k] = candidate[k];
            MHcounts_gamma[k] += 1.;
        } else {
            candidate[k] = gamma[k];
        }
    }
    //return true;
    //delete [] candidate;
}
    
        /*
    double loglike_old = gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, gamma);
    double loglike_new =gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, candidate);
    double alpha = std::min(1.0, exp(loglike_new - loglike_old));
    
    if (gen.uniform() < alpha) {
        //logger.log("spline coeffs updated.");
        for (size_t k=0; k<L; ++k) {
            gamma[k] = candidate[k];
        }
        MHcounts_gamma += 1.;
        return true;
    } else {
        //logger.log("spline coeffs unchanged.")
        return false;
    }
}*/


// try the collapsed Gibbs algorithm
void gammaMHcons(double rho, double eps, int degree, size_t ngrids, double stepsize, double* MHcounts_gamma, const std::vector<double>& event, const std::vector<double>& yobs, double* allfit, const std::vector<double>& knots, double* gamma, RNG& gen) {
    
    size_t L = (knots.size() - degree - 1);
    size_t n = event.size();
    double* candidate = new double[L];
    for (size_t k=0; k<L; ++k) {
        candidate[k] = gamma[k];
    }
    double sum_iden = 0.;
    double extra = 0.;
    double extra_0 = 0.;
    arma::mat Sig_e(L, L, arma::fill::zeros);
    arma::mat penalMat(L-2, L, arma::fill::zeros);
    arma::vec mu_e(L, arma::fill::zeros);
    for (size_t k=0; k<(L-2); k++) {
        penalMat(k,k) = 1.;
        penalMat(k,k+1) = -2.;
        penalMat(k,k+2) = 1.;
    }
    
    for (size_t j=0; j<n; ++j) {
        for (size_t k=0; k<L; ++k) {
            for (size_t l=0; l<L; ++l) {
                Sig_e(k,l) += std::max(0., bspline(yobs[j], k, degree, knots))*std::max(0., bspline(yobs[j], l, degree, knots));
            }
        }
    }
    
    Sig_e = Sig_e + penalMat.t() * penalMat * rho;
    Sig_e = inv(Sig_e);
    
    
    // univariate Metropolis steps
    for (size_t k=1; k<L; ++k) {
    //for (size_t k=1; k<L; ++k) {
        /*
        arma::vec e = rmvnorm_post(mu_e, Sig_e);
        sum_iden = 0.;
        for (size_t l=0; l<L; l++) {
            sum_iden += e[l]*e[l];
        }
        for (size_t l=0; l<L; l++) {
            e[l] = e[l] / sqrt(sum_iden);
        }*/
        sum_iden = 0.;
        
        //candidate[k] = gamma[k] + stepsize*gen.normal()*e[k];
        
        for (size_t m=0; m < event.size(); ++m) {
            extra += std::max(0., bspline(yobs[m], k, degree, knots));
            extra_0 += std::max(0., bspline(yobs[m], 0, degree, knots));
        }
        
        for (size_t l=1; l<L; ++l) {
            for (size_t m=0; m < event.size(); ++m) {
                sum_iden += gamma[l]*std::max(0., bspline(yobs[m], l, degree, knots));
            }
        }
        
        sum_iden -= gamma[k]*extra;
        sum_iden -= gamma[0]*extra_0;
        
        
        candidate[0] = gamma[0] + stepsize*gen.normal();
        candidate[k] = (-sum_iden - candidate[0]*extra_0) / extra;
        
        double loglike_old = gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, gamma);
        double loglike_new = gammaFC(rho, eps, degree, ngrids, event, yobs, allfit, knots, candidate);
        double alpha = std::min(1.0, exp(loglike_new - loglike_old));
        
        if (gen.uniform() < alpha) {
            gamma[k] = candidate[k];
            MHcounts_gamma[k] += 1.;
            if (k == (L-1)) {
                gamma[0] = candidate[0];
                MHcounts_gamma[0] += 1.;
            }
        } else {
            candidate[k] = gamma[k];
            if (k == (L-1)) {
                candidate[0] = gamma[0];
            }
        }
    }
    //return true;
    //delete [] candidate;
    
}
