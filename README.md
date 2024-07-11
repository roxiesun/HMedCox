# Code for HMedCox
Authors: Rongqian Sun and Xinyuan Song

Paper link: tbc

Contact: <sunrq@link.cuhk.edu.hk>

This directory provides implementation of heteroeneous mediation analysis for Cox proportional hazards model with multiple mediators with simulated dataset. The code regarding binary trees are constructed following the framework of R package [bcf](https://github.com/jaredsmurray/bcf) and the work of [Hahn et al. (2020)](https://projecteuclid.org/journals/bayesian-analysis/volume-15/issue-3/Bayesian-Regression-Tree-Models-for-Causal-Inference--Regularization-Confounding/10.1214/19-BA1195.full) on Bayesian causal forest.

## Implementation details
```
packages <- c('Rcpp', 'RcppArmadillo','survival', 'survminer')
lapply(packages, require, character.only = TRUE)
source('bthm.cpp')
## or
# system("R CMD INSTALL bthm_4.0.7.tar.gz")
# packages <- c('Rcpp', 'RcppArmadillo','survival', 'survminer','bthm')
# lapply(packages, require, character.only = TRUE)
```
- To run the algorithm, simply input the time-to-event outcome, status, covariate matrix, mediator matrix, and binary treatment to the R function BTHM and specify the number of iterations. A toy example is given below and further details can be found in simulation.R.
```
n = 1000
p = 25
p_m = 2
nhcut = 10 
n_cmntsample = 1

covmat = matrix(1, nrow = p, ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    covmat[i,j] = 0.3^abs(i-j) + 0.1*(i!=j)
  }
}
x = rmvnorm(n, sigma = covmat)
g = abs(x[,4] - 1)
q =  -1 + g + abs(x[,3]-1)

# generate treatment variable
ps = 0.8*pnorm(0.2*q/sd(q) - 0.1*x[,1]) + 0.05 + runif(n)/10
a = rbinom(n,1,ps)
  

tau_m2 =  -1 + 0.5*x[,7] - 0.25*log(x[,5]^2)
tau_m1 =  -1 + 0.5*x[,2] - 0.5*abs(x[,7])
tau_y =   -(-0.5 + 1*abs(x[,6]) - sqrt(abs((x[,6] - 2)*x[,2])))
mu1 = 0.15*(x[,4] - 2)^2 + abs(x[,3] - 1)
mu2 = 0.5*q + cos(pi*x[,9])

Sigma = matrix(c(1.0,0.1,0.1,1.44), nrow = 2)
m.true = cbind(mu1 + tau_m1*a, mu2 + tau_m2*a) + rmvnorm(n,sigma = Sigma)

temp = exp( -0.2*abs(x[,3]) + 0.5*(x[,12])
            + 1*m.true[,2]*(1 + 0.5*(x[,6] > 0.5)) + 1*m.true[,1]  - 1*tau_y*a )
truehzd = log(temp)
T <- (-1/(1*temp))*log(runif(n))

C <- runif(n, min = 0.5, max = quantile(T,0.9)) 
yobs <- (T <= C)*T + (T > C)*C
delta <- ifelse(T <= C, 1, 0)
  
pihat = glm(a ~ x, family = "binomial"(link = "probit"))$fitted.values
cat('Number of failed: ', sum(delta), ' .\n')
cat('Censoring rate: ', sum(1-delta)*100/n, '% .\n')  
 
fit <- bthm::BTHM(y = yobs, status = delta, a = a, m=m.true, x_control = x, pihat = pihat, 
                  nburn =1000, nsim = 2000, verbose = TRUE, truehzd = truehzd,  G_ph = nhcut, nknots = 10, vs = TRUE)
```
- Outputs:
  + posterior draws of the direct effects: fit$tau
  + variable usage count for control/modifer tree: fit$scon, fit$smod, fit$scoxcon
