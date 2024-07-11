#include <string>
#include <vector>
#include <map>
#include "tree.h"

using std::string;
using std::cout;
using std::endl;

// constructors
tree::tree(): p_mu(1), v(0), c(0), p(0), l(0), r(0) {mu.resize(p_mu);}
tree::tree(std::vector<double>& m, size_t& pmu): mu(m), p_mu(pmu), v(0), c(0), p(0), l(0), r(0) {}
tree::tree(const tree& n): p_mu(1), v(0), c(0), p(0), l(0), r(0) {mu.resize(p_mu);cp(this, &n);}

// operators
tree& tree::operator=(const tree& rhs)
{
    if(&rhs != this){
        tonull(); //kill left hand side
        cp(this, &rhs); // copy the rhs to lhs
    }
    return *this;
}
//-----------------------------------------------
//output / input vector (for mu)
//template <typename S>
std::ostream& operator<<(std::ostream& os,
                    const std::vector<double>& vector)
{
    // Printing all the elements
    // using <<
    //os << "[";
    os << vector.size()<< " ";
    for (int i = 0; i < vector.size(); ++i) {
            os << vector[i];
            if (i != vector.size() - 1)
                os << " ";
        }
    //os << "]";
    return os;
}
std::istream& operator>>(std::istream& is, std::vector<double>& vector)
{
    size_t vs;
    vector.resize(0);
    
    is >> vs;
    if (!is) {
        Rcpp::Rcout << ">> error: unreadable size of mu" << endl;
        return is;
    }
    double temp;
    //std::fill(vector.begin(), vector.end(), 0);
    for (int i = 0; i < vs; ++i){
        is >> temp;
        vector.push_back(temp);
    }
    if (vector.size() != vs) {
        Rcpp::Rcout << ">> error: vector size unmatched" << endl;
        return is;
    }
    return is;
}

// output a tree
/* access each node and output its info:
 id? var? cutoff? mu?*/
std::ostream& operator<<(std::ostream& os, const tree& t)
{
    tree::cnpv nds;
    t.getnodes(nds);
    os << nds.size() << endl;
    for(size_t i = 0; i<nds.size();i++){
        os << nds[i]->nid() << " ";
        os << nds[i]->getv() << " ";
        os << nds[i]->getc() << " ";
        os << nds[i]->getp_mu() << " ";
        os << nds[i]->getmu() << endl;
    }
    return os;
}
//-------------------------------------------------

// input a tree
std::istream& operator>>(std::istream& is, tree& t)
{
    size_t tid, pid; /*tid is id of current node; pid is id of parents*/
    std::map<size_t, tree::tree_p> pts; // pointers to nodes index by node id
    size_t nn; // #nodes
    t.tonull(); // wipe old tree
    
    // read in number of nodes
    is >> nn;
    if (!is) {
        Rcpp::Rcout << ">> error: unreadable number of nodes" << endl;
        return is;
    }
    // read in vector of node info
    std::vector<node_info> nv(nn);
    for(size_t i=0; i!=nn; i++){
        is >> nv[i].id >> nv[i].v >> nv[i].c >> nv[i].p_mu >> nv[i].mu;
        if (!is) {
            Rcpp::Rcout << ">> error: unreadable node info on node" << i+1 << endl;
            return is;
        }
    }
    
    // set first node to the top
    pts[1] = &t; /*Note: pts[1] is not the first pointer, but the poinyer of id1*/
    t.setv(nv[0].v); t.setc(nv[0].c); t.setp_mu(nv[0].p_mu); t.setmu(nv[0].mu);
    t.p = 0; // the top has no parent
    
    // take loop through the rest nodes
    for(size_t i=1; i!=nv.size();i++){
        tree::tree_p np = new tree;
        np->v = nv[i].v; np->c = nv[i].c; np->p_mu = nv[i].p_mu; np->mu = nv[i].mu;
        tid = nv[i].id; // id of current node
        pts[tid] = np; //update the map
        pid = tid/2; // round down？！parent id
        //set pointers
        if(tid % 2 == 0) { //left child
            pts[pid]->l = np;
        } else{
            pts[pid]->r = np;
        }
        np->p = pts[pid];
        /* now the node parameter and rule has been set(mu,v,c), and tree strcuture related to this node is added (p,l,r)*/
    }
    return is;
}
//-------------------------------------------------

// output cutpoint info Xinfo
std::ostream& operator<<(std::ostream& os, const xinfo& xi)
{
    os << xi.size() << endl;
    for (size_t i=0; i<xi.size(); i++) {
        os << xi[i].size() << endl;
        for (size_t j = 0; j < xi[i].size(); j++)
            os << xi[i][j] << endl;
        os << endl; // print by row?
    }
    return os;
}
//-------------------------------------------------

// input cutpoint info Xinfo
std::istream& operator>>(std::istream& is, xinfo& xi)
{
    size_t xin;  // size of Xinfo, the 2d array, nrow*ncol?
    size_t vecdn; // size of each row/var?
    
    xi.resize(0); // reset old xinfo
    
    is >> xin;
    if (!is) {
        Rcpp::Rcout << ">> error: unreadable size of Xinfo" << endl;
        return is;
    }
    
    std::vector<double> vec_d;
    double vecdelem;
    for (size_t i=0; i<xin; i++) {
        is >> vecdn;
        for (size_t j=0; j<vecdn; j++) {
            is >> vecdelem; // input Xinfo[i][j]
            vec_d.push_back(vecdelem);
        }
        xi.push_back(vec_d); // add the i_th var's info at back
        vec_d.resize(0);
    }
    return is;
}
//-------------------------------------------------



// public functions
// find leaf nodes given x and splitting rules
/* Note: the tree class is acturally in a sense of both tree and node, double*x actually points to the start of a covariate vector, and we are comparing one of the element in Xinfo based on a single node with v and c predefined, so this is the Recursion*/
tree::tree_cp tree::bot(double *x, xinfo& xi)
{
    if (l==0) return this; // bottom node
    if (x[v] < xi[v][c]) {
        return l->bot(x, xi);
    } else {
        return r->bot(x, xi);
    }
}
//-------------------------------------------------

//find region for a given variable
void tree::region(size_t v, int *L, int *U)
{
    if (p == 0) { //no parent
        return; // end here
    }
    if (p->v == v) { // does my parent also use v
        if (this == p->l) { // am I the left child or right
            if ((int)(p->c) <= (*U)) *U = (p->c)-1;
        } else {
            if ((int)(p->c) >= (*L)) *L = (p->c)+1;
        }
    }
    p->region(v, L, U); //Recursion again
}

void tree::region_l(size_t v, int *L)
{
    if (l==0) { // no children
        return;
    }
    
    if (this->v == v && (int)(this->c) >= (*L)) {
        *L = (int)(this->c)+1;
        r->region_l(v, L);
    }
    else {
        l->region_l(v, L);
        r->region_l(v, L);
    }
}

void tree::region_u(size_t v, int* U)
{
    if (l==0) {
        return;
    }
    
    if (this->v == v && (int)(this->c) <= (*U)) {
        *U = (int)(this->c)-1;
        l->region_u(v, U);
    }
    else {
        l->region_u(v, U);
        r->region_u(v, U);
    }
}

/*is this node in a right subtree that splits on var v*/
bool tree::isrightofv(size_t v)
{
    if (p==0)  // root
        return false;
    else if (p->v != v)
        return p->isrightofv(v);
    else if (p->r == this)
        return true;
    return false;
}

bool tree::isleftofv(size_t v)
{
    if (p==0)  // root
        return false;
    else if (p->v != v)
        return p->isleftofv(v);
    else if (p->l == this)
        return true;
    return false;
}

// swap the left and right branches of this node
void tree::swaplr()
{
    tree_p temp;
    temp = this->r;
    this->r = this->l;
    this->l = temp;
}
//-------------------------------------------------

size_t tree::nuse(size_t v)
{
    npv nds;
    this->getnodes(nds); // get all nodes
    size_t nu = 0;
    for (size_t i=0; i!=nds.size(); i++) {
        if (nds[i]->l && nds[i]->v==v) nu+=1;
    }
    return nu;
}


void tree::varsplits(std::set<size_t> &splits, size_t v)
{
    npv nds;
    this->getnodes(nds);
    for (size_t i=0; i!=nds.size(); i++) {
        if (nds[i]->l && nds[i]->v==v) {
            splits.insert(nds[i]->c); // add c to splitting rule
        }
    }
}
//-------------------------------------------------

// tree functions
size_t tree::treesize() const
{
    if (l==0) return 1; // bottom
    else return (1 + l->treesize() + r->treesize());
}

size_t tree::nnogs() const
{
    if (!l) return 0; // bottom nodes
    if (l->l || r->l) { // not a nog
        return (l->nnogs() + r->nnogs());
    } else { // is a nog
        return 1;
    }
}

size_t tree::nbots() const
{
    if (l==0) {
        return 1;
    } else {
        return l->nbots() + r->nbots();
    }
}

// node functions
size_t tree::depth() const{
    if (!p) return 0;
    else return (1+p->depth());
}

size_t tree::nid() const
{
    if (!p) return 1;
    if (this==p->l) return 2*(p->nid());
    else return 2*(p->nid())+1;
}

char tree::ntype() const
{
    if (!p) return 't'; //Note: t can also be b
    if (!l) return 'b';
    if (!(l->l) && !(r->l)) return 'n';
    return 'i';
}

tree::tree_p tree::getptr(size_t nid){
    if (this->nid() == nid) return this;
    if (l==0) return 0; /// not found this id
    tree_p lp = l->getptr(nid);
    if (lp) return lp; // found on left subtree
    tree_p rp = r->getptr(nid);
    if (rp) return rp; // found on right
    return 0; // not found
}

bool tree::isnog() const
{
    bool isnog = true;
    if (l) {
        if (l->l || r->l) isnog = false;
    } else {
        isnog = false; // no child, bottom is not nog node
    }
    return isnog;
}

bool tree::isleft() const
{
    bool isleft = false;
    if (p && p->l==this)
        isleft = true;
    
    return isleft;
}

bool tree::isright() const
{
    bool isright = false;
    if (p && p->r==this) {
        isright = true;
    }
    return isright;
}

void tree::tonull()
{
    size_t ts = treesize();
    while (ts>1) { // is false, ts = 1, only top node
        npv nv;
        getnogs(nv);
        for (size_t i=0; i<nv.size(); i++) {
            delete nv[i]->l;
            delete nv[i]->r;
            nv[i]->l=0;
            nv[i]->r=0;
        }
        ts = treesize();
    }
    mu.resize(1);//mu = 0.0; p_mu = 1;
    v = 0; c = 0;
    p = 0; l = 0; r = 0; p_mu = 1;
}

//-------------------------------------------------
// get node pointer vectors
void tree::getbots(npv& bv){
    if (l) {
        l->getbots(bv);
        r->getbots(bv);
    } else {
        bv.push_back(this);
    }
}

void tree::getnogs(npv& nv)
{
    if (l) {
        if (l->l || r->l) {
            if (l->l) l->getnogs(nv);
            if (r->l) r->getnogs(nv);
        } else {
            nv.push_back(this);
        }
    }
}

void tree::getintnodes_nov(npv &nv, size_t var)
{
    if (l) {
        if (this->v != var)
            nv.push_back(this);
        l->getintnodes_nov(nv,var);
        r->getintnodes_nov(nv,var);
    }
}

void tree::getnodes_onv(npv &v, size_t var)
{
    if (this->v == var)
        v.push_back(this);
    if (l) {
        l->getnodes_onv(v,var);
        r->getnodes_onv(v,var);
    }
}

void tree::getnodes_onvc(npv &v, size_t var, size_t cut)
{
    if (this->v==var && this->c==cut)
        v.push_back(this);
    if (l) {
        l->getnodes_onvc(v, var, cut);
        r->getnodes_onvc(v, var, cut);
    }
}

void tree::getnodes(npv &v)
{
    v.push_back(this);
    if (l) {
        l->getnodes(v);
        r->getnodes(v);
    }
}

void tree::getnodes(cnpv &v) const
{
    v.push_back(this);
    if (l) {
        l->getnodes(v);
        r->getnodes(v);
    }
}

void tree::getnbots(npv &v)
{
    if (this->l) {
        v.push_back(this);
        this->l->getnbots(v);
        if (this->r->l)
            this->r->getnbots(v);
    }
}

// get nodes except bare roots and leafs
void tree::getrotnodes(npv &v)
{
    if (!this->p && this->l) { // the root node do have children
        this->l->getnbots(v);
        this->r->getnbots(v);
    }
}

void tree::getrotelems(npv &v)
{
    if (this->l) { // has child
        if (this->v != this->p->v)
            v.push_back(this);
        this->l->getrotelems(v);
        if (this->r->l)
            this->r->getrotelems(v);
    }
}

void tree::getswap(npv& v)
{
    if (this->l) { // has children
        if (!(this->isnog())) { // has grandchild (child has rule)
            v.push_back(this);
            this->l->getswap(v);
            this->r->getswap(v);
        }
    }
}
//-------------------------------------------------

// birth: add nodes
bool tree::birth(size_t nid, size_t v, size_t c, std::vector<double>& mul, std::vector<double>& mur)
{
    tree_p np = getptr(nid); //get the node pointer with id = nid
    if (np==0) {
        Rcpp::Rcout << "error in birth: bottom node not found\n";
        return false; // not found node with this id
    }
    if (np->l) {
        Rcpp::Rcout << "error in birth: found node has already have children\n";
        return false;
    }
    
    // otherwise add children to bottom
    tree_p l = new tree;
    l->mu = mul; l->p_mu = mul.size();
    tree_p r = new tree;
    r->mu = mur; r->p_mu = mur.size();
    np->l = l;
    np->r = r;
    np->v = v; np->c = c;
    l->p = np;
    r->p = np;
    
    return true;
}

// death: delete children
bool tree::death(size_t nid, std::vector<double>& mu)
{
    tree_p nb = getptr(nid);
    if (nb==0) {
        Rcpp::Rcout << "error in death: invalid nid\n";
        return false;
    }
    // delete the kids only if the nid node has no grandchild
    if (nb->isnog()) {
        delete nb->l;
        delete nb->r;
        nb->l=0;
        nb->r=0;
        nb->v=0;
        nb->c=0;
        nb->mu=mu; nb->p_mu = mu.size(); // the exact mu when nid turns to a bottom node
        return true;
    } else {
        Rcpp::Rcout << "error in death: node is not a nog node\n";
        return false;
    }
}


//-------------------------------------------------
// private functions

void tree::cp(tree_p n, tree_cp o)
{
    if (n->l) {
        Rcpp::Rcout << "cp error: node has children\n";
        return;
    }
    
    n->mu = o->mu; n->p_mu = o->p_mu;
    if (n->p_mu != n->mu.size()) {
      Rcpp::Rcout << "cp error: mu size(" << n->mu.size() <<") and p_mu(" << n->p_mu <<") not match\n";
    }
    n->v = o->v;
    n->c = o->c;
    
    if (o->l) {
        n->l = new tree;
        (n->l)->p = n;
        cp(n->l, o->l);
        (n->r)->p = n;
        cp(n->r, o->r);
    }
}

// add children to bottom node *np
void tree::birthp(tree_p np, size_t v, size_t c, std::vector<double>& mul, std::vector<double>& mur)
{
    tree_p l = new tree;
    l->mu = mul; l->p_mu = mul.size();
    tree_p r = new tree;
    r->mu = mur; r->p_mu = mur.size();
    np->l = l;
    np->r = r;
    np->v = v; np->c = c;
    l->p = np;
    r->p = np;
}

// kill children of nog node *nb and update value of mu
void tree::deathp(tree_p nb, std::vector<double>& mu)
{
    delete nb->l;
    delete nb->r;
    nb->l = 0;
    nb->r = 0;
    nb->v = 0;
    nb->c = 0;
    nb->mu = mu; nb->p_mu = mu.size();
}
//-------------------------------------------------
// print the tree node by node
void tree::pr(xinfo &xi) const
{
    size_t d = depth();
    size_t id = nid();
    
    size_t pid;
    if (!p) pid = 0;
    else pid = p->nid();
    string pad(2*d,' ');
    string sp(", ");
    if (ntype()=='t')
        Rcpp::Rcout << "tree size: " << treesize() << endl;
    
    Rcpp::Rcout << pad << "id: " << id;
    Rcpp::Rcout << sp << "var index: " << v;
    Rcpp::Rcout << sp << "cut index: " << c;
    if (ntype()=='b' || treesize() == 1) {
        Rcpp::Rcout << sp << "threshold: N/A";
    } else {
        Rcpp::Rcout << sp << "threshold: " << xi[v][c];
    }
    Rcpp::Rcout << sp << "mu size: " << p_mu;
    Rcpp::Rcout << sp << "mu: " << mu;
    Rcpp::Rcout << sp << "type: " << ntype();
    Rcpp::Rcout << sp << "depth: " << depth();
    Rcpp::Rcout << endl;
    
    if (l) {
        l->pr(xi);
        r->pr(xi);
    }
}

void tree::pr() const
{
    size_t d = depth();
    size_t id = nid();
    
    size_t pid;
    if (!p) pid = 0;
    else pid = p->nid();
    
    string pad(2*d, ' ');
    string sp(", ");
    if (ntype()=='t')
        Rcpp::Rcout << "tree size: " << treesize() << endl;
    
    Rcpp::Rcout << pad << "id: " << id;
    Rcpp::Rcout << sp << "var index: " << v;
    Rcpp::Rcout << sp << "cut index: " << c;
    Rcpp::Rcout << sp << "threshold: unavailable";
    Rcpp::Rcout << sp << "mu size: " << p_mu;
    Rcpp::Rcout << sp << "mu: " << mu;
    Rcpp::Rcout << sp << "type: " << ntype();
    Rcpp::Rcout << sp << "depth: " << depth();
    Rcpp::Rcout << endl;
    
    if (l) {
        l->pr();
        r->pr();
    }
    
}
