// set tree class and iostream to read/write the trees and cutpoints
#ifndef GUARD_tree_h
#define GUARD_tree_h

#include<iostream>
#include<fstream>
#include<cmath>
#include<cstddef>
#include<vector>

#include "info.h"
#include "rand_gen.h"
#include "logging.h"
#include "ProbHypers.h"

// node info
struct node_info{
    std::size_t id; // node id
    std::size_t v; // variable
    std::size_t c; //cut point
    //double *mu; // mu
    std::size_t p_mu; // dim of mu
    std::vector<double> mu;
};
/*Note: 3 ways to access a node:
 1. its id
 2. the index into the array of node pointers, returned by class function getnodes or getbots or get nogs
 3. its pointer (const pointer)
 */

// xinfo[v][c] is the c_th cutpoint of v_th variable
// split rule: left if x[v] < xinfo[v][c]

typedef std::vector<double> vec_d;
typedef std::vector<vec_d> xinfo; //2d array, split rules

class tree{
public:
    // typedefs first
    typedef tree* tree_p; // tree pointer
    typedef const tree* tree_cp; /* pointer to const,You can modify tree_cp itself but the object pointed to by tree_cp shall not be modified */
    typedef std::vector<tree_p> npv; // node point vector
    typedef std::vector<tree_cp> cnpv; //const node point vector
    
    
    // friend functions
    friend std::istream& operator>>(std::istream&, tree&); /* overloading operator >> to facilitate writing*/
    friend std::istream& operator>>(std::istream& is, std::vector<double>& vector);
    friend void getLU(tree::tree_p pertnode, xinfo& xi, int* L, int* U);
    friend void getpertLU(tree::tree_p pertnode, size_t pertvar, xinfo& xi, int* L, int* U);
    friend void getpertsuff(tree& x, tree::tree_p pertnode, tree::npv& bnv, size_t oldc, size_t oldvar, bool didswaplr, xinfo& xi, dinfo& di, std::vector<sinfo>& svold, std::vector<sinfo>& svnew);
    friend void getinternalvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars);
    friend int getnumcuts(tree::tree_p n, xinfo& xi, size_t var);
    friend void rotright(tree::tree_p n);
    friend void rotleft(tree::tree_p n);
    friend void reduceleft(tree::tree_p n, size_t v, size_t c);
    friend void reduceright(tree::tree_p n, size_t v, size_t c);
    friend bool isequalvc(tree::tree_p t1, tree::tree_p t2);
    friend bool isequal(tree::tree_p t1, tree::tree_p t2);
    friend void splitleft(tree::tree_p t, size_t v, size_t c);
    friend void splitright(tree::tree_p t, size_t v, size_t c);
    friend bool splitsonv(tree::tree_p t, size_t v);
    friend bool splitsonv(tree::tree_p nl, tree::tree_p nr, size_t v);
    friend void nwaysonsplit(tree::tree_p tl, tree::tree_p tr, int *nways);
    friend bool hasvcsplit(tree::tree_p t, size_t v, size_t c);
    friend bool isleaf(tree::tree_p t);
    friend bool arenodesequal(tree::tree_p nl, tree::tree_p nr);
    friend bool arenodesleafs(tree::tree_p nl, tree::tree_p nr);
    friend bool merge(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c, int* nways, RNG& gen);
    friend bool canmerge(tree::tree_p tl, tree::tree_p tr, size_t v, size_t c, int* nways);
    friend bool merge2(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c, RNG& gen);
    friend bool merge3(tree::tree_p tl, tree::tree_p tr, size_t v, size_t c, int* nways);
    friend bool merge2nt(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c, int* nways, int atway, RNG& gen);
    friend bool merge3nt(tree::tree_p tl, tree::tree_p tr, size_t v, size_t c, int* nways);
    friend bool merge2triv(tree::tree_p tl, tree::tree_p tr, tree::tree_p t, size_t v, size_t c);

 #ifdef MPIBART
    friend bool bd(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
    friend void pertcv(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
    friend void chgv(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
    friend bool rjbd(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
    friend bool rotp(tree::tree_p tnew, tree& x, xinfo& xi, pinfo& pi, RNG& gen, tree::npv& rnodes, size_t numslaves);
 #else
    //my functions
    friend bool bd(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
    friend bool bd_lg(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior);
    friend bool bd_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);
    friend bool bd_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, std::vector<double>& probs, ProbHypers& hypers);
    friend bool draw_prior_withvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers);
    friend bool draw_prior_lgwithvs(tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt, arma::mat Covsig, arma::mat Covprior, bool vs, ProbHypers& hypers);
    friend void GenBelow(tree::tree_p x, pinfo& pi, dinfo& di, xinfo& xi, double* phi, ProbHypers& hypers, RNG& gen, bool vs);
    friend bool rotphet(tree::tree_p tnew, tree& x, xinfo& xi, dinfo& di, double* phi, pinfo& pi, RNG& gen);
    //end my functions
    friend void pertcv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void chgv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend bool rjbd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend bool rotp(tree::tree_p tnew, tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    //friend bool changerule(tree& x, xinfo& xi, dinfo& di, double*phi, pinfo& pi, RNG& gen, Logger logger, std::vector<size_t>& ivcnt);
 #endif

 #ifdef CALIBART
 #ifdef MPIBART
    friend bool bd(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
    friend void calipertcv(tree& x, xinfo& xi, pinfo& pi, RNG& gen, size_t numslaves);
 #elif CALIBIAS
    friend bool calibiasbd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void calibiaspertcv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void calibiaschgv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void getcalibiasgoodvars(tree::tree_p n, xinfo& xi, pinfo& pi, std::vector<size_t>& goodvars);
    friend bool calibiasrotp(tree::tree_p tnew, tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
 #else
    friend bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void calipertcv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend void calichgv(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
    friend bool calirotp(tree::tree_p tnew, tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen);
 #endif
 #endif
    
    tree(); //constructor
    tree(const tree&);
    tree(std::vector<double>&, size_t&);
    ~tree() {tonull();} // destructor
    
    // operator
    tree& operator=(const tree&);
    
    // node access
    void setmu(std::vector<double>& mu) {this->mu = mu;}
    void setv(size_t v) {this->v = v;}
    void setc(size_t c) {this->c = c;}
    void setp_mu(size_t p_mu) {this->p_mu = p_mu;}
    
    std::vector<double> getmu() const {return mu;}
    size_t getv() const {return v;}
    size_t getc() const {return c;}
    tree_p getp() const {return p;} // get parent node
    tree_p getl() const {return l;}
    tree_p getr() const {return r;}
    size_t getp_mu() const {return p_mu;}
    
    // tree functions
    // ------------------
    // charac of a tree
    size_t treesize() const; // # nodes
    size_t nnogs() const; // # no grandchild nodes
    size_t nbots() const; // # leaf nodes
    void pr(xinfo& xi) const;
    void pr() const; // screen and print?
    
    // birth and death using node id
    bool birth(size_t nid, size_t v, size_t c, std::vector<double>& mul, std::vector<double>& mur);
    bool death(size_t nid, std::vector<double>& mu);
    
    // get node pointer vectors
    void getbots(npv& bv);
    void getnogs(npv& nv);
    void getintnodes_nov(npv& nv, size_t var); /* get all internal nodes not splitted on this var*/
    void getnodes(npv& v);// get all nodes
    void getnodes(cnpv& v) const;
    void getnodes_onv(npv& v, size_t var); /*get all nodes splitted on this var*/
    void getnodes_onvc(npv& v, size_t var, size_t cut); /*get all nodes splitted on this var and this cut*/
    void getnbots(npv& v); // all except leafs
    void getrotnodes(npv& v); // rot nodes?
    void getrotelems(npv& v);
    void getswap(npv& v);
    
    // find node w.r.t X and splitting rules
    tree_cp bot(double *x, xinfo& xi); /*find leaf for a given x*/
    void region(size_t v, int* L, int* U); /*find region for a var*/
    void region_l(size_t v, int* L);
    void region_u(size_t v, int* U);
    bool isrightofv(size_t v); /*is this node in a right subtree that splits on var v*/
    bool isleftofv(size_t v);
    void swaplr();
    size_t nuse(size_t v); // #number of use in rule
    void varsplits(std::set<size_t> &splits, size_t v);
    //--------------------------------------
    
    // node functions
    // -------------------------------
    size_t depth() const;
    size_t nid() const;
    char ntype() const; /* t-top, b-bottom, n-nog,i-interior Note: t can be b*/
    tree_p getptr(size_t nid); // get node pointer
    bool isnog() const;
    bool isleft() const;
    bool isright() const;
    void tonull();
    //--------------------------------
    
private:
    // node parameter
    //double *mu;
    vec_d mu;
    size_t p_mu;
    // rule
    size_t v;
    size_t c;
    
    // tree struc
    tree_p p; // parent
    tree_p l; // left child
    tree_p r; // right child
    
    // utility funcs
    void cp(tree_p n, tree_cp o); // copy tree
    void birthp(tree_p np, size_t v, size_t c, std::vector<double>& mul, std::vector<double>& mur);
    void deathp(tree_p nb, std::vector<double>& mu);
};

std::istream& operator>>(std::istream&, tree&);
std::ostream& operator<<(std::ostream&, const tree&);
std::istream& operator>>(std::istream&, xinfo&);
std::ostream& operator<<(std::ostream&, const xinfo&);
//template <typename S>
std::ostream& operator<<(std::ostream& os,
                         const std::vector<double>& vector);
//template <typename S>
std::istream& operator>>(std::istream& is, std::vector<double>& vector);

#endif /* tree_h */
