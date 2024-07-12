#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)
print(args)

nrep = as.numeric(args[2])
flag = as.numeric(args[1])

startseed = (flag - 1)*nrep

#if()

packages <- c("Rcpp", "mvtnorm", "RcppArmadillo", "survival", 
              "extraDistr", "abind", "tidyverse", "viridis", "ggplot2", "latex2exp")


lapply(packages, require, character.only = TRUE)

library(bthm)


# data generating process
n = 1000
p = 25
p_m = 2
nhcut = 10 
n_cmntsample = 1


time.begin <- proc.time()
for (irep in 1:nrep) {
  set.seed(startseed + irep - 1)
  
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
  
  
  tau_m1 =  -1 + 0.5*x[,2] - 0.5*abs(x[,7])
  tau_m2 =  -1 + 0.5*x[,7] - 0.25*log(x[,5]^2)
  tau_y =   -(-0.5 + 1*abs(x[,6]) - sqrt(abs((x[,6] - 2)*x[,2])))
  
  mu1 = 0.15*(x[,4] - 2)^2 + abs(x[,3] - 1)
  mu2 = 0.5*q + cos(pi*x[,9])
  
  Sigma = matrix(c(1.0,0.3,0.3,1.0), nrow = 2)
  m.true = cbind(mu1 + tau_m1*a, mu2 + tau_m2*a) + rmvnorm(n,sigma = Sigma)
  Sigma = matrix(c(1.0,0.0,0.0,1.0), nrow = 2)
  
  
  #censoring time and survival time
  Dat2 <- cbind(x, m.true)
  
  temp = exp( -0.2*abs(x[,3]) + 0.5*(x[,12]) 
               + 1*m.true[,2]*(1 + 0.5*(x[,6] > 0.5)) + 1*m.true[,1]  - 1*tau_y*a )
  truehzd = log(temp)
  T <- (-1/(1*temp))*log(runif(n))
 
  
  C <- runif(n, min = 0.5, max = quantile(T,0.9)) # if 0.5t + 0.2
  
  yobs <- (T <= C)*T + (T > C)*C
 
  delta <- ifelse(T <= C, 1, 0)
  
  pihat = glm(a ~ x, family = "binomial"(link = "probit"))$fitted.values
  cat('Number of failed: ', sum(delta), ' .\n')
  cat('Censoring rate: ', sum(1-delta)*100/n, '% .\n')
  
  # by default, varselection = false
  testfit <- bthm::BTHM(y = yobs, status = delta, a = a, m=m.true, x_control = x, pihat = pihat, 
                  nburn =1000, nsim = 2000, verbose = TRUE, truehzd = truehzd,  G_ph = nhcut, nknots = 10, vs = TRUE, flag = flag, 
                  irep = irep)
  
  
  
  
  
  
  IE1 = 1*tau_m1
  IE2 = 1*tau_m2*(1 + 0.5*(x[,6] > 0.5))
  IE = IE1 + IE2
  DE = -tau_y
  TE = IE1 + IE2 + DE

  
  cat('Comonotone sampling started.\n')
  #comonotone sampling
  nsim = dim(testfit$m_ctr)[1]
  nsample = n_cmntsample
  n1 = sum(a)
  
  jntsample_m0 = array(NA, c(n, p_m, nsim*nsample))
  for(t in 1:nsim){
    for (s in 1:nsample) {
      jntsample_m0[,,(t-1)*nsample + s] = t(apply(testfit$m_ctr[t,,], 1, function(x) rmvnorm(1, mean = x, sigma = matrix(testfit$covsigma[t,], nrow = p_m))))
        
    }
  }
  
  U1 = U2 = array(NA, c(n, nsim*nsample))
  for(t in 1:nsim) {
      for (s in 1:nsample) {
        U1[, (t-1)*nsample + s] = pnorm(jntsample_m0[,1,(t-1)*nsample + s], mean = testfit$m_ctr[t,,1], sd = rep(sqrt(testfit$sigma[t,1]), n))
        U2[, (t-1)*nsample + s] = pnorm(jntsample_m0[,2,(t-1)*nsample + s], mean = testfit$m_ctr[t,,2], sd = rep(sqrt(testfit$sigma[t,2]), n))
        
    }
  }
  
  jntsample_m1 = array(NA, c(n, p_m, nsim*nsample))
  for (t in 1:nsim) {
    for (s in 1:nsample) {
      jntsample_m1[,1, (t-1)*nsample + s] = qnorm(U1[, (t-1)*nsample + s], mean = testfit$m_trt[t,,1], sd = rep(sqrt(testfit$sigma[t,1]), n))
      jntsample_m1[,2, (t-1)*nsample + s] = qnorm(U2[, (t-1)*nsample + s], mean = testfit$m_trt[t,,2], sd = rep(sqrt(testfit$sigma[t,2]), n))
    }
  }
  
  
  posttry0 = bthm::Ypostpred.bthm(testfit, x_pred_coxcon = x, x_pred_moderate = x, 
                                  m_post = jntsample_m0, pihat_pred = pihat, ndraws = nsample, 
                                  save_tree_directory = getwd(), flag = flag, irep = irep)
  posttry1 = bthm::Ypostpred.bthm(testfit, x_pred_coxcon = x, x_pred_moderate = x, 
                                  m_post = jntsample_m1, pihat_pred = pihat, ndraws = nsample, 
                                  save_tree_directory = getwd(), flag = flag, irep = irep)
  
  
  cfsample_m01 = abind(jntsample_m0[,1,,drop=FALSE], jntsample_m1[,2,,drop = FALSE], along = 2)
  cfsample_m10 = abind(jntsample_m1[,1,,drop=FALSE], jntsample_m0[,2,,drop = FALSE], along = 2)
  
  posttry_m01 = bthm::Ypostpred.bthm(testfit, x_pred_coxcon = x, x_pred_moderate = x, 
                                  m_post = cfsample_m01, pihat_pred = pihat, ndraws = nsample, 
                                  save_tree_directory = getwd(), flag = flag, irep = irep)
  
  posttry_m10 = bthm::Ypostpred.bthm(testfit, x_pred_coxcon = x, x_pred_moderate = x, 
                                     m_post = cfsample_m10, pihat_pred = pihat, ndraws = nsample, 
                                     save_tree_directory = getwd(), flag = flag, irep = irep)
  
  
  cat('Comonotone sampling done.\n')
  
  
  
  # with lambda0(t) = 1, Lambda0(t) = T
  Survfunc.true <- function(t, a, x, m){
  
    tau_y =  -(-0.5 + 1*abs(x[,6]) - sqrt(abs((x[,6] - 2)*x[,2])))
    s = exp(-(t) * exp(-0.2*abs(x[,3]) + 0.5*(x[,12]) 
                                       + m[,2]*(1 + 0.5*(x[,6] > 0.5)) + 1*m[,1]  - 1*tau_y*a))
    return (s)
  }
  
  
  
  jointm0.true = array(NA, c(n, p_m, nsim*nsample))
  for (t in 1:nsim) {
    for (s in 1:nsample) {
      jointm0.true[,,(t-1)*nsample + s] = t(apply(cbind(mu1, mu2), 1, function(x) rmvnorm(1, mean = x, sigma = Sigma)))
    }
  }
  
  U1.true = U2.true = array(NA, c(n, nsim*nsample))
  for(t in 1:nsim) {
    for (s in 1:nsample) {
      U1.true[, (t-1)*nsample + s] = pnorm(jointm0.true[,1,(t-1)*nsample + s], mean = mu1, sd = rep(sqrt(Sigma[1,1]), n))
      U2.true[, (t-1)*nsample + s] = pnorm(jointm0.true[,2,(t-1)*nsample + s], mean = mu2, sd = rep(sqrt(Sigma[2,2]), n))
      
    }
  }
  
  
  jointm1.true = array(NA, c(n, p_m, nsim*nsample))
  for (t in 1:nsim) {
    for (s in 1:nsample) {
      jointm1.true[,1, (t-1)*nsample + s] = qnorm(U1.true[, (t-1)*nsample + s], mean = array(mu1 + tau_m1), sd = rep(sqrt(Sigma[1,1]), n))
      jointm1.true[,2, (t-1)*nsample + s] = qnorm(U2.true[, (t-1)*nsample + s], mean = array(mu2 + tau_m2), sd = rep(sqrt(Sigma[2,2]), n))
    }
  }
  
  
  
  #combined marginal m samples for the interventional effect
  margm01.true = abind(jointm0.true[,1,,drop = FALSE], jointm1.true[,2,,drop = FALSE], along = 2)
  margm10.true = abind(jointm1.true[,1,,drop = FALSE], jointm0.true[,2,,drop = FALSE], along = 2)
  
  
  
  s111.true = rowMeans(apply(jointm1.true, 3, Survfunc.true, t = mean(yobs[delta==1]), a = rep(1,n), x = x)) 
  s000.true = rowMeans(apply(jointm0.true, 3, Survfunc.true, t = mean(yobs[delta==1]), a = rep(0,n), x = x)) 
  s011.true = rowMeans(apply(jointm1.true, 3, Survfunc.true, t = mean(yobs[delta==1]), a = rep(0,n), x = x)) 
  s001.true = rowMeans(apply(margm01.true, 3, Survfunc.true, t = mean(yobs[delta==1]), a = rep(0,n), x = x)) 
  s010.true = rowMeans(apply(margm10.true, 3, Survfunc.true, t = mean(yobs[delta==1]), a = rep(0,n), x = x)) 
  
  
  Survfunc.est <- function(hpost, ypost){
    s = exp(-as.vector(hpost) * exp(ypost))
    return (s)
  }
  
  s111.est = colMeans(Survfunc.est(testfit$h0_survp[,1],  posttry1$pred_y1))
  s000.est = colMeans(Survfunc.est(testfit$h0_survp[,1],  posttry0$pred_y0))
  s011.est = colMeans(Survfunc.est(testfit$h0_survp[,1],  posttry1$pred_y0))
  s001.est = colMeans(Survfunc.est(testfit$h0_survp[,1],  posttry_m01$pred_y0))
  s010.est = colMeans(Survfunc.est(testfit$h0_survp[,1],  posttry_m10$pred_y0))
  
  
 
  s111.post = (Survfunc.est(testfit$h0_survp[,1],  posttry1$pred_y1))
  s000.post = (Survfunc.est(testfit$h0_survp[,1],  posttry0$pred_y0))
  s011.post = (Survfunc.est(testfit$h0_survp[,1],  posttry1$pred_y0))
  s001.post = (Survfunc.est(testfit$h0_survp[,1],  posttry_m01$pred_y0))
  s010.post = (Survfunc.est(testfit$h0_survp[,1], posttry_m10$pred_y0))
  
  
  
  ## aggregating the monte carlo samples for mediation formula
  y111_post = array(unlist(aggregate(posttry1$pred_y1, list(rep(1:(nrow(posttry1$pred_y1) %/% nsample + 1),
                                                                each = nsample, len = nrow(posttry1$pred_y1))), mean)[-1]), dim = c(nsim, n))
  
  y011_post = array(unlist(aggregate(posttry1$pred_y0, list(rep(1:(nrow(posttry1$pred_y0) %/% nsample + 1),
                                                                each = nsample, len = nrow(posttry1$pred_y0))), mean)[-1]), dim = c(nsim, n))
  
  y010_post = array(unlist(aggregate(posttry_m10$pred_y0, list(rep(1:(nrow(posttry_m10$pred_y0) %/% nsample + 1),
                                                                   each = nsample, len = nrow(posttry_m10$pred_y0))), mean)[-1]), dim = c(nsim, n))
  
  y000_post = array(unlist(aggregate(posttry0$pred_y0, list(rep(1:(nrow(posttry0$pred_y0) %/% nsample + 1),
                                                                each = nsample, len = nrow(posttry0$pred_y0))), mean)[-1]), dim = c(nsim, n))
  
  y001_post = array(unlist(aggregate(posttry_m01$pred_y0, list(rep(1:(nrow(posttry_m01$pred_y0) %/% nsample + 1),
                                                                   each = nsample, len = nrow(posttry_m01$pred_y0))), mean)[-1]), dim = c(nsim, n))
  
  
  
  
  # on Survival probability
  s111_post = array(unlist(aggregate(s111.post, list(rep(1:(nrow(s111.post) %/% nsample + 1),
                                                         each = nsample, len = nrow(s111.post))), mean)[-1]), dim = c(nsim, n))
  
  s011_post = array(unlist(aggregate(s011.post, list(rep(1:(nrow(s011.post) %/% nsample + 1),
                                                         each = nsample, len = nrow(s011.post))), mean)[-1]), dim = c(nsim, n))
  
  s010_post = array(unlist(aggregate(s010.post, list(rep(1:(nrow(s010.post) %/% nsample + 1),
                                                         each = nsample, len = nrow(s010.post))), mean)[-1]), dim = c(nsim, n))
  
  s000_post = array(unlist(aggregate(s000.post, list(rep(1:(nrow(s000.post) %/% nsample + 1),
                                                         each = nsample, len = nrow(s000.post))), mean)[-1]), dim = c(nsim, n))
  
  s001_post = array(unlist(aggregate(s001.post, list(rep(1:(nrow(s001.post) %/% nsample + 1),
                                                         each = nsample, len = nrow(s001.post))), mean)[-1]), dim = c(nsim, n))
  
  
  
  # Draw the boxplot and the histogram 
  my_variable=cbind(colMeans(y111_post - y011_post), colMeans(y010_post - y000_post),
                    colMeans(y001_post - y000_post), colMeans(y011_post - y000_post), 
                    colMeans(y111_post - y000_post))
  
  
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  # Draw the boxplot and the histogram 
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,1] , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,1] , ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,1], plot = FALSE)
  hist_2 = hist(DE, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(y111_post - y011_post), lwd = 2)
  abline(v = mean(DE), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,2] , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,2] , ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow M_1 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,2], plot = FALSE)
  hist_2 = hist(IE1, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow M_1 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(y010_post - y000_post), lwd = 2)
  abline(v = mean(IE1), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,3] , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,3] , ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow M_2 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,3], plot = FALSE)
  hist_2 = hist(IE2, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow M_2 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(y001_post - y000_post), lwd = 2)
  abline(v = mean(IE2), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,4] , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,4] , ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow MT}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,4], plot = FALSE)
  hist_2 = hist(IE, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, A \\rightarrow MT}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(y011_post - y000_post), lwd = 2)
  abline(v = mean(IE), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,5] , horizontal=TRUE , xaxt="n" , col=rgb(0.8,0.8,0,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,5] , ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, total}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,5], plot = FALSE)
  hist_2 = hist(TE, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.2,0.8,0.5,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{\\lambda}_{i, total}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(y111_post - y000_post), lwd = 2)
  abline(v = mean(TE), lwd = 2, col = "blue")
  
  
  my_variable=cbind(colMeans(s111_post - s011_post), colMeans(s010_post - s000_post),
                    colMeans(s001_post - s000_post), colMeans(s011_post - s000_post), 
                    colMeans(s111_post - s000_post))
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,1] , horizontal=TRUE , xaxt="n" , col=rgb(0.7,0.9,0.2,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,1] , ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  #ax = pretty((min(c(my_variable[,1], s111.true - s011.true)) - 0.001):max(c(my_variable[,1], s111.true - s011.true)), n = 12)
  hist_1 = hist(my_variable[,1], plot = FALSE)
  hist_2 = hist(s111.true - s011.true, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(s111_post - s011_post), lwd = 2)
  abline(v = mean(s111.true - s011.true), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,2] , horizontal=TRUE , xaxt="n" , col=rgb(0.7,0.9,0.2,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,2] , ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow M_1 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,2], plot = FALSE)
  hist_2 = hist(s010.true - s000.true, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow M_1 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(s010_post - s000_post), lwd = 2)
  abline(v = mean(s010.true - s000.true), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,3] , horizontal=TRUE , xaxt="n" , col=rgb(0.7,0.9,0.2,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,3] , ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow M_2 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,3], plot = FALSE)
  hist_2 = hist(s001.true - s000.true, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow M_2 \\rightarrow T}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(s001_post - s000_post), lwd = 2)
  abline(v = mean(s001.true - s000.true), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,4] , horizontal=TRUE , xaxt="n" , col=rgb(0.7,0.9,0.2,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,4] , ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow MT}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,4], plot = FALSE)
  hist_2 = hist(s011.true - s000.true, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, A \\rightarrow MT}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(s011_post - s000_post), lwd = 2)
  abline(v = mean(s011.true - s000.true), lwd = 2, col = "blue")
  
  layout(mat = matrix(c(1,2),2,1, byrow=TRUE),  height = c(1,8))
  
  par(mar=c(0, 3.1, 1.1, 2.1) + 0.1)
  boxplot(my_variable[,5] , horizontal=TRUE , xaxt="n" , col=rgb(0.7,0.9,0.2,0.5) , frame=F)
  par(mar=c(6, 3.1, 1.1, 2.1) + 0.1)
  #hist(my_variable[,5] , ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, total}$"), cex.lab = 1.5, cex.axis = 1.5)
  hist_1 = hist(my_variable[,5], plot = FALSE)
  hist_2 = hist(s111.true - s000.true, plot = FALSE)
  plot(hist_1, ylab = "", col=rgb(0.1,0.9,0.2,0.5) , border=F,  main="" , xlab=TeX("$\\hat{\\Omega}^{S}_{i, total}$"), cex.lab = 1.5, cex.axis = 1.5)
  plot(hist_2, col=rgb(0.6,0.1,0.9,0.0), add = TRUE)
  abline(v = mean(s111_post - s000_post), lwd = 2)
  abline(v = mean(s111.true - s000.true), lwd = 2, col = "blue")
  
  
  file.remove(paste("./con_trees.",1,".", flag,".", irep, ".txt", sep = ""))
  file.remove(paste("./mod_trees.",1,".", flag,".", irep, ".txt", sep = ""))
  file.remove(paste("./coxcon_trees.",1,".", flag,".", irep, ".txt", sep = ""))
}
time.total <- proc.time() - time.begin

gc()


