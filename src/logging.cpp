#include <RcppArmadillo.h>
#include "logging.h"
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::plugins("cpp11")]]

using namespace Rcpp;

Logger::Logger(){  // constructor
    level = 0;
    depth = 0;
}

void Logger::setLevel(int inpLevel){
    level = inpLevel;
}

void Logger::startContext(){
    depth +=1;
}
void Logger::stopContext(){
    depth += -1;
}

void Logger::getVectorHead(Rcpp::NumericVector x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f ...", x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
}

void Logger::getVectorHead(std::vector<double> x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f ...", x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
}

void Logger::getVectorHead(double* x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f ...", x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
}

void Logger::getVectorHead(std::vector<int> x, char s[100]){
    std::sprintf(s,"%d, %d, %d, %d, %d, %d, %d ...", x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
}

void Logger::log(std::string text){
    if (level > 0) {
        for (int didx = 0; didx < depth; didx++) {
            Rcout << "--";
        }
        if (depth > 0) {
            Rcout << " ";
        }
        Rcout << text << "\n";
    }
}
