#ifndef GUARD_logging_h
#define GUARD_logging_h
#include <RcppArmadillo.h>

// is this for logger file? system recording?
class Logger{
private:
    int level;  // for the tree?
    int depth;
    
public:
    Logger(); // constructor
    
    void log(std::string text);
    void setLevel(int inpLevel);
    void startContext();
    void stopContext();
    void getVectorHead(Rcpp::NumericVector x, char s[100]);
    void getVectorHead(std::vector<double> x, char s[100]);
    void getVectorHead(double* x, char s[100]);
    void getVectorHead(std::vector<int> x, char s[100]);
};

#endif /* logging_h */
