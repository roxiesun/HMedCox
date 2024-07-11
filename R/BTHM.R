#' @importFrom stats approxfun lm qchisq quantile sd
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom abind abind

Rcpp::loadModule(module = "TreeSamples", TRUE)

# Note: the code is established based on R package bcf @ https://github.com/jaredsmurray/bcf

.cp_quantile = function(x, num = 1000, cat_levels = 8){
  nobs = length(x)
  nuniq = length(unique(x))

  if(nuniq == 1){
    ret = x[1]
    warning("A supplied covariate contains a single unique value.")
  } else if(nuniq < cat_levels) {
    xx = sort(unique(x))
    ret = xx[-length(xx)] + diff(xx)/2
  } else {
    q = approxfun(sort(x), quantile(x, p=0:(nobs-1)/nobs))
    ind = seq(min(x),max(x),length.out = num)
    ret = q(ind)
  }
  return(ret)
}


.get_chain_tree_files = function(tree_path, chain_id, arg1 = 1, arg2 = 1){
  out <- list("con_trees" = paste0(tree_path, '/',"con_trees.", chain_id, ".", arg1, ".", arg2, ".txt"),
              "mod_trees" = paste0(tree_path, '/',"mod_trees.", chain_id, ".", arg1, ".", arg2, ".txt"),
              "coxcon_trees" = paste0(tree_path, '/',"coxcon_trees.", chain_id, ".", arg1, ".", arg2, ".txt"))
  return(out)
}

.get_do_type = function(n_cores){
  if(n_cores>1){
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)
    `%doType%`  <- foreach::`%dopar%`
  } else {
    cl <- NULL
    `%doType%`  <- foreach::`%do%`
  }

  do_type_config <- list('doType'  = `%doType%`,
                         'n_cores' = n_cores,
                         'cluster' = cl)

  return(do_type_config)
}

.cleanup_after_par = function(do_type_config){
  if(do_type_config$n_cores>1){
    parallel::stopCluster(do_type_config$cluster)
  }
}

.ident <- function(...){
  # courtesy https://stackoverflow.com/questions/19966515/how-do-i-test-if-three-variables-are-equal-r
  args <- c(...)
  if( length( args ) > 2L ){
    #  recursively call ident()
    out <- c( identical( args[1] , args[2] ) , .ident(args[-1]))
  }else{
    out <- identical( args[1] , args[2] )
  }
  return( all( out ) )
}

# The following function is referenced from
# https://github.com/nchenderson/AFTrees
# density of inv-chisq
.dinvchisq <- function(x, nu){
  ans <- (nu/2)*log(nu/2) - lgamma(nu/2) - (nu/2 + 1)*log(x) - nu/(2*x)
  return(exp(ans))
}


.pchiconv <- function(p,nu){
  ## CDF for the random variable defined as nu/X + Z, X ~ chisq_nu, Z ~ N(1,1)
  ff <- function(v,x){
    ans <- pnorm(x-v, mean = 1, sd = 1)*.dinvchisq(v, nu)
    return(ans)
  }
  tau <- length(p)
  a <- rep(0, tau)
  for (k in 1:tau) {
    a[k] <- integrate(ff, x = p[k], lower = 0, upper = Inf)$value
  }
  return(a)
}

.qchiconv <- function(q, nu){
  gg <- function(x){
    .pchiconv(x, nu) - q
  }
  a <- uniroot(gg, interval = c(0,200))
  return(a$root)
}

.FindKappa <- function(q, sighat, nu){
  if(q < 0.905 & q > 0.895){
    Q <- 6.356677
  } else if (q < 0.995 & q > 0.985) {
    Q <- 27.17199
  } else if (q < .8 & q > .7){
    Q <- 3.858311
  } else if (q < .55 & q > .45){
    Q <- 2.531834
  } else if (q < .3 & q > .2){
    Q <- 1.569312
  } else {
    Q <- .qchiconv(q, nu)
  }
  Kappa <- sighat^2/Q
  return(Kappa)
}


#' @useDynLib bthm
#' @export
BTHM <- function(y, status, a, m, x_control, x_moderate = x_control, x_coxcon = x_control,
                pihat, w = NULL, random_seed = sample.int(.Machine$integer.max,1),
                n_chains = 1, n_cores = n_chains,n_threads = max((RcppParallel::defaultNumThreads()-2)/n_cores, 1),
                nburn = 500, nsim = 1000, nthin = 1, update_interval = 100,
                ntree_control = 200,
                #sd_control = 2*sd(y),
                base_control = 0.95,
                power_control = 2,
                ntree_moderate = 100,
                #sd_moderate = sd(y),
                base_moderate = 0.95,
                power_moderate = 3,
                ntree_coxcon = 200,
                #sd_cure = 0.5, # as a try
                base_coxcon = 0.95,
                power_coxcon = 2,
                nknots = 20,
                G_ph = 5,
                t_survp = mean(y[status==1]),
                truehzd = NULL,
                save_tree_directory = '.',
                nu = 3, sigq = 0.99, sighat = NULL,
                include_pi = "control", verbose = TRUE, vs = FALSE, flag = 1, irep = 1) {

  if(is.null(w)){
    w <- matrix(1,ncol = 1,nrow = length(y))
  }
  
  if(is.null(truehzd)){
    truehzd <- matrix(0, ncol = 1, nrow = length(unique(y)))
  }

  pihat = as.matrix(pihat)

  ### TODO range check on parameters

  x_c = matrix(x_control, ncol = ncol(x_control))
  x_m = matrix(x_moderate, ncol = ncol(x_moderate))
  x_cox = matrix(x_coxcon, ncol = ncol(x_coxcon))
  # include ps as a covariate
  if(include_pi=="both" | include_pi=="control") {
    x_c = cbind(x_c, pihat)
  }
  
  if(include_pi=="both" | include_pi=="moderate") {
    x_m = cbind(x_m, pihat)
  }
  
  if(include_pi=="both" | include_pi == "cox" | include_pi == "control") {
    x_cox = cbind(x_cox, pihat)
  }
  
  
  ## include m to x_cox
  if (nrow(x_cox) == nrow(m)) {
    x_cox = cbind(x_cox, m)
  } else { cat("error: x_cox and m dim mismatch.\n")}
  
  ## include a to x_cox
  # x_cox = cbind(x_cox, a)

  ## set cutpoints
  cutpoint_list_c = lapply(1:ncol(x_c), function(i) .cp_quantile(x_c[,i]))
  cutpoint_list_m = lapply(1:ncol(x_m), function(i) .cp_quantile(x_m[,i]))
  cutpoint_list_cox = lapply(1:ncol(x_cox), function(i) .cp_quantile(x_cox[,i]))

  ## make sigest for mediator equations
  #sighat = apply(m,2,sd)
  #zeta = 2*sighat
  
  

  #kappa <- .FindKappa(sigq, sighat[1], nu)
  if(is.null(sighat)) {
    sighat = c()
    kappa = c()
    for (i in 1:ncol(m)) {
      lmf = lm(m[,i]~a+as.matrix(x_c), weights = w)
      sighat = c(sighat,summary(lmf)$sigma)
      #sd(y) #summary(lmf)$sigma
    }
    qchi = qchisq(1.0-sigq,nu)
    kappa = c(kappa,(sighat*sighat*qchi)/nu)
  }
  zeta = 8*sighat


  dir = tempdir()
  perm = order(a, decreasing = TRUE) # so that the treated are displayed first
  
  hcsig_coxcon = c()
  hcsig_mod = c()
  for (i in 1:(nburn + nthin*nsim + 1)) {
    hcsig_coxcon = c(hcsig_coxcon, rhcauchy(1, 1.5/sqrt(ntree_coxcon)))
    hcsig_mod = c(hcsig_mod, rhcauchy(1, 0.5/sqrt(ntree_moderate)))
  }

   
  intv = quantile(y, seq(0,1,1/G_ph))
  intv[1] = 0
   
  RcppParallel::setThreadOptions(numThreads = n_threads)

  do_type_config <- .get_do_type(n_cores)
  `%doType%` <- do_type_config$doType


  chain_out <- foreach::foreach(iChain = 1:n_chains,
                                .export = c(".get_chain_tree_files","BTHMRcpp")) %doType% {
    this_seed = random_seed + iChain - 1
    cat("Calling BTHMRcpp From R \n")
    set.seed(this_seed)

    tree_files = .get_chain_tree_files(save_tree_directory, iChain, arg1 = flag, arg2 = irep)

    print(tree_files)

    fit_bthm = BTHMRcpp(y_ = y[perm], a_ = a[perm], w_ = w[perm],
                                 x_con_ = t(x_c[perm,,drop=FALSE]), x_mod_ = t(x_m[perm,,drop=FALSE]), 
                                 x_coxcon_ = t(x_cox[perm,,drop = FALSE]), m_ = t(m[perm,, drop = FALSE]),
                                 status_ = status[perm],
                                 x_con_info_list = cutpoint_list_c,
                                 x_mod_info_list = cutpoint_list_m,
                                 x_coxcon_info_list = cutpoint_list_cox, intv_ = intv,
                                 burn = nburn, nd = nsim, thin = nthin, ntree_coxcon = ntree_coxcon,
                                 ntree_mod = ntree_moderate, ntree_con = ntree_control,
                                 nu = nu,
                                 #con_sd = con_sd,
                                 #mod_sd = mod_sd,
                                 con_alpha = base_control,
                                 con_beta = power_control,
                                 mod_alpha = base_moderate,
                                 mod_beta = power_moderate,
                                 coxcon_alpha = base_coxcon,
                                 coxcon_beta = power_coxcon,
                                 kappa = kappa,
                                 sigest_ = sighat,
                                 zeta_ = zeta,
                                 truehzd_ = truehzd[perm],
                                 hcsig_coxcon_ = hcsig_coxcon,
                                 hcsig_mod_ = hcsig_mod,
                                 treef_con_name_ = tree_files$con_trees,
                                 treef_mod_name_ = tree_files$mod_trees,
                                 treef_coxcon_name_ = tree_files$coxcon_trees,
                                 t_survp = t_survp,
                                 printevery = update_interval,
                                 nknots = nknots,
                                 verbose_sigma = verbose,
                                 vs = vs)

    cat("BTHMRcpp returned to R\n")
    
    coxconfit <- fit_bthm$coxcon_post[, order(perm)]
    
    confit0 <- fit_bthm$con_post
    
    confit <- list()
    dim_m = ncol(m)
    for (i in 1:dim_m) {
      mat = cbind(confit0[, i])
      for (j in 1:(n-1)) {
        mat = cbind(mat, confit0[, j*dim_m+i])
      }
      confit[[i]] = mat[, order(perm)]
    }
    
    modfit0 <- fit_bthm$b_post
    modfit <- list()
    tau_post <- list()
    for (i in 1:dim_m) {
      mat = cbind(modfit0[, i])
      for (j in 1:(n-1)) {
        mat = cbind(mat, modfit0[, j*(dim_m)+i]) # or, dim_m +1
      }
      modfit[[i]] = mat[, order(perm)]*(1.0/(fit_bthm$b1 - fit_bthm$b0))
      tau_post[[i]] = mat[, order(perm)]
    }
    #mat = cbind(modfit0[, (dim_m + 1)]) # or, dim_m +1
   # for (j in 1:(n-1)) {
  #    mat = cbind(mat, modfit0[, j*(dim_m+1)+dim_m+1]) # or, dim_m +1
  #  }
    #modfit[[dim_m+1]] = mat[, order(perm)]
    tau_y_post = fit_bthm$coxtau_post[, order(perm)]

    #ac = fitbcf$m_post[, order(perm)] # stores allfit_con

    #Tm = fitbcf$b_post[, order(perm)]*(1.0/(fitbcf$b1 - fitbcf$b0)) # stores the allfit_mod/bscale, the pure tree fit

    #Tc = ac*(1.0/fitbcf$msd) #stores allfit_con/mscale, the pure tree fit

    #tau_post = fitbcf$b_post[, order(perm)]
    
    m_ctr_post = array(NA, c(nsim, nrow(m), ncol(m)))
    m_trt_post = array(NA, c(nsim, nrow(m), ncol(m)))
    for (i in 1:dim_m) {
      m_ctr_post[,,i] = confit[[i]] + modfit[[i]]*fit_bthm$b0
        #confit[[i]] + modfit[[i]]*fit_bthm$b0
      m_trt_post[,,i] = confit[[i]] + modfit[[i]]*fit_bthm$b1
        #confit[[i]] + modfit[[i]]*fit_bthm$b1
    }

    
    y_ctr_post = coxconfit
    y_trt_post = coxconfit + tau_y_post
    
    #m1hat_post = fit_bthm$mhat_post[, seq(1, length(m),2)]
    #m2hat_post = fit_bthm$mhat_post[, seq(2, length(m),2)]
    #m1hat_post = m1hat_post[, order(perm)]
    #m2hat_post = m2hat_post[, order(perm)]
    for (i in 1:dim_m) {
      assign(paste("m",i,"hat_post", sep = ""), (fit_bthm$mhat_post[, seq(i, length(m),dim_m)])[, order(perm)])
      assign(paste("tau",i,"_post", sep = ""), (fit_bthm$b_post[, seq(i, length(m),dim_m)])[, order(perm)])
      
    }
    
    #tau1_post = fit_bthm$b_post[, seq(1, length(m),2)]
    #tau2_post = fit_bthm$b_post[, seq(2, length(m),2)]
    #tau1_post = tau1_post[, order(perm)]
    #tau2_post = tau2_post[, order(perm)]



    #mu_post = null_inter + Tc*fitbcf$msd + Tm*fitbcf$b0 #stores E[Y|0,X] for everyone, note: without intercept

    #mu_post_trt = null_inter + Tc*fitbcf$msd + Tm*fitbcf$b1 # stores E[Y|1,X] for everyone note: without intercept

    varcnt_con_post = fit_bthm$varcnt_con_post
    varcnt_mod_post = fit_bthm$varcnt_mod_post
    varcnt_coxcon_post = fit_bthm$varcnt_coxcon_post
    
    # mhat_post0 = fit_bthm$mhat_post
    # mhat_post <- list()
    # for (i in 1:dim_m) {
    #   mat = cbind(mhat_post0[, i])
    #   for (j in 1:(n-1)) {
    #     mat = cbind(mat, mhat_post0[, j*dim_m+i])
    #   }
    #   mhat_post[[i]] = mat[, order(perm)]
    # }
    
    #mhat_post = array(NA, c(dim(m1hat_post), ncol(m)))
    #tauhat_post = array(NA, c(dim(tau1_post), ncol(m)+1))
    mhat_post = abind(m1hat_post, along = 3)
    tauhat_post = abind(tau1_post, along = 3)
    
    if (dim_m > 1) {
      for(i in 2:(dim_m)) {
        mhat_post = abind(mhat_post, get(paste("m",i,"hat_post", sep = "")))
        tauhat_post = abind(tauhat_post, get(paste("tau",i,"_post", sep = "")))
      }
    }
    
    tauhat_post = abind(tauhat_post, tau_y_post)
    
    #mhat_post[,,1] = m1hat_post
    #mhat_post[,,2] = m2hat_post
    #tauhat_post[,,1] = tau1_post
    #tauhat_post[,,2] = tau2_post
    #tauhat_post[,,3] = tau_y_post


    #chain_out = list()
    #chain_out[[1]] <-
    list(sigma = fit_bthm$sigma_post,
         h0 = fit_bthm$hzd0_post[,order(perm)],
         h0_survp = fit_bthm$hzdsurvp_post,
         yhat = fit_bthm$yhat_post[,order(perm)],
         mhat = mhat_post,
         #muy = null_inter,
         m_ctr  = m_ctr_post, # list
         m_trt = m_trt_post, # list
         #tau_m = tau_post, # list
         #mu_scale = fitbcf$msd,
         tau = tauhat_post,
         #tau_m_scale = fit_bthm$bsd,
         y_ctr = y_ctr_post,
         y_trt = y_trt_post,
         #tau_y = tau_y_post,
         b0 = fit_bthm$b0,
         b1 = fit_bthm$b1,
         perm = perm,
         include_pi = include_pi,
         random_seed=this_seed,
         varcnt_con = varcnt_con_post,
         varcnt_mod = varcnt_mod_post,
         varcnt_coxcon = varcnt_coxcon_post,
         covsigma = fit_bthm$covsigma_post,
         ##adrch = fit_bthm$adrch_post,
         scon = fit_bthm$scon_post,
         smod = fit_bthm$smod_post,
         scoxcon = fit_bthm$scoxcon_post,
         rddt = fit_bthm$rddt_post
    )
  }

  all_sigma = c()
  all_covsigma = c()
  all_h0 = c()
  all_h0_survp = c()
  all_yhat = c()
  all_mhat = c()
  
  all_m_ctr = c()
  all_m_trt = c()
  all_tau = c()

  all_b0 = c()
  all_b1 = c()

  all_y_ctr   = c()
  all_y_trt   = c()
  

  all_varcnt_con = c()
  all_varcnt_mod = c()
  all_varcnt_coxcon = c()
  
  ##all_adrch = c()
  all_scon = c()
  all_smod = c()
  all_scoxcon = c()
  all_rddt = c()


  chain_list=list()

  #n_iter = length(chain_out[[1]]$sigma)
  #
  for (iChain in 1:n_chains){
    sigma <- chain_out[[iChain]]$sigma
    h0 <- chain_out[[iChain]]$h0
    h0_survp <- chain_out[[iChain]]$h0_survp
    yhat <- chain_out[[iChain]]$yhat
    mhat <- chain_out[[iChain]]$mhat
    #mu_scale  <- chain_out[[iChain]]$mu_scale
    #tau_scale <- chain_out[[iChain]]$tau_scale
    
    m_ctr  <- chain_out[[iChain]]$m_ctr
    m_trt  <- chain_out[[iChain]]$m_trt
    
    y_ctr  <- chain_out[[iChain]]$y_ctr
    y_trt  <- chain_out[[iChain]]$y_trt

    b0 <- chain_out[[iChain]]$b0
    b1 <- chain_out[[iChain]]$b1

    tau <- chain_out[[iChain]]$tau
    
    varcnt_con <- chain_out[[iChain]]$varcnt_con
    varcnt_mod <- chain_out[[iChain]]$varcnt_mod
    varcnt_coxcon <- chain_out[[iChain]]$varcnt_coxcon
    covsigma <- chain_out[[iChain]]$covsigma
    
    ##adrch <- chain_out[[iChain]]$adrch
    scon <- chain_out[[iChain]]$scon
    smod <- chain_out[[iChain]]$smod
    scoxcon <- chain_out[[iChain]]$scoxcon
    rddt <- chain_out[[iChain]]$rddt
    

    # -----------------------------
    # Support Old Output
    # -----------------------------
    all_sigma = rbind(all_sigma, sigma)
    all_h0 = rbind(all_h0, h0)
    all_h0_survp = rbind(all_h0_survp, h0_survp)
    #all_mu_scale = c(all_mu_scale,  mu_scale)
    #all_tau_scale = c(all_tau_scale, tau_scale)
    all_b0 = c(all_b0, b0)
    all_b1 = c(all_b1, b1)

    all_yhat = rbind(all_yhat, yhat)
    all_mhat = abind(all_mhat, mhat, along = 1)
    
    all_m_ctr  = abind(all_m_ctr, m_ctr, along = 1)
    all_m_trt  = abind(all_m_trt, m_trt, along = 1)
    all_y_ctr  = rbind(all_y_ctr, y_ctr)
    all_y_trt  = rbind(all_y_trt, y_trt)
    all_tau = abind(all_tau, tau, along = 1)

    all_varcnt_con = rbind(all_varcnt_con,varcnt_con)
    all_varcnt_mod = rbind(all_varcnt_mod,varcnt_mod)
    all_varcnt_coxcon = rbind(all_varcnt_coxcon, varcnt_coxcon)
    all_covsigma = rbind(all_covsigma, covsigma)
    
    ##all_adrch = rbind(all_adrch, adrch)
    all_scon = rbind(all_scon, scon)
    all_smod = rbind(all_smod, smod)
    all_scoxcon = rbind(all_scoxcon, scoxcon)
    all_rddt = c(all_rddt, rddt)
    
    all_rddt_post = matrix(rep(all_rddt, nrow(m)), nrow = nsim)
    #all_yhat = all_yhat - all_rddt_post
    #all_y_ctr  = all_y_ctr - all_rddt_post
    #all_y_trt  = all_y_trt - all_rddt_post
    #all_h0 = all_h0 * exp(all_rddt_post[,1:ncol(all_h0)])


    # -----------------------------
    # Make the MCMC Object
    # -----------------------------

    scalar_df <- data.frame("sigma" = sigma,
                            "h0" = h0,
                            "tau_bar" = apply(tau, c(1,3), mean),#matrixStats::rowWeightedMeans(tau, w),
                            "y_ctr_bar"  = matrixStats::rowWeightedMeans(y_ctr, w),
                            "y_trt_bar"  = matrixStats::rowWeightedMeans(y_trt, w),
                            "m_ctr_bar"  = apply(m_ctr, c(1,3), mean),#matrixStats::rowWeightedMeans(m_ctr, w),
                            "m_trt_bar"  = apply(m_trt, c(1,3), mean),#matrixStats::rowWeightedMeans(m_trt, w),
                            "yhat_bar" = matrixStats::rowWeightedMeans(yhat, w),
                            "mhat_bar" = apply(mhat, c(1,3), mean),#matrixStats::rowWeightedMeans(mhat, w),
                            #"mu_scale" = mu_scale,
                            #"tau_scale" = tau_scale,
                            "b0"  = b0,
                            "b1"  = b1)
    



    chain_list[[iChain]] <- coda::as.mcmc(scalar_df)
    # -----------------------------
    # Sanity Check Constants Accross Chains
    # -----------------------------
    #if(chain_out[[iChain]]$con_sd   != chain_out[[1]]$con_sd)     stop("con_sd not consistent between chains for no reason")
    #if(chain_out[[iChain]]$mod_sd   != chain_out[[1]]$mod_sd)     stop("mod_sd not consistent between chains for no reason")
    #if(chain_out[[iChain]]$muy      != chain_out[[1]]$muy)        stop("muy not consistent between chains for no reason")
    if(chain_out[[iChain]]$include_pi != chain_out[[1]]$include_pi) stop("include_pi not consistent between chains for no reason")
    if(any(chain_out[[iChain]]$perm   != chain_out[[1]]$perm))      stop("perm not consistent between chains for no reason")
  }

  fitObj <- list(sigma = all_sigma,
                 h0 = all_h0,
                 h0_survp = all_h0_survp,
                 yhat = all_yhat,
                 mhat = all_mhat,
                 #sdy = chain_out[[1]]$sdy,
                 #muy = chain_out[[1]]$muy,
                 y_ctr  = all_y_ctr,
                 y_trt  = all_y_trt,
                 m_ctr  = all_m_ctr,
                 m_trt  = all_m_trt,
                 tau = all_tau,
                 #mu_scale = all_mu_scale,
                 #tau_scale = all_tau_scale,
                 b0 = all_b0,
                 b1 = all_b1,
                 perm = perm,
                 include_pi = chain_out[[1]]$include_pi,
                 random_seed = chain_out[[1]]$random_seed,
                 coda_chains = coda::as.mcmc.list(chain_list),
                 raw_chains = chain_out,
                 varcnt_con = all_varcnt_con,
                 varcnt_mod = all_varcnt_mod,
                 varcnt_coxcon = all_varcnt_coxcon,
                 dim_m = ncol(m),
                 covsigma = all_covsigma,
                 ##adrch = all_adrch,
                 scon = all_scon,
                 smod = all_smod,
                 scoxcon = all_scoxcon,
                 rddt = all_rddt)

  attr(fitObj, "class") <- "bthm"

  .cleanup_after_par(do_type_config)

  return(fitObj)
}

# TO BE Modified
#' @export Mpredict.bthm
#' @export
Mpredict.bthm <- function(object, x_pred_control, x_pred_moderate, #x_pred_coxcon, 
                           pihat_pred, #a_pred,
                           save_tree_directory, flag = 1, irep = 1, ncores = 1, 
                           ...) {
  if(any(is.na(x_pred_control))) stop("Missing values in x_pred_control")
  if(any(is.na(x_pred_moderate))) stop("Missing values in x_pred_moderate")
  #if(any(is.na(x_pred_coxcon))) stop("Missing values in x_pred_coxcon")
  if(any(is.na(pihat_pred))) stop("Missing values in pihat_pred")
  #if(any(is.na(a_pred))) stop("Missing values in a_pred")

  pihat_pred = as.matrix(pihat_pred)

  x_pc = matrix(x_pred_control, ncol = ncol(x_pred_control))
  x_pm = matrix(x_pred_moderate, ncol = ncol(x_pred_moderate))
  #x_pcox = matrix(x_pred_cure, ncol = ncol(x_pred_coxcon))

  if(object$include_pi == "both" | object$include_pi == "control") {
    x_pc = cbind(x_pc, pihat_pred)
  }

  if(object$include_pi == "both" | object$include_pi == "moderate") {
    x_pm = cbind(x_pm, pihat_pred)
  }
  
  # if(object$include_pi == "both" | object$include_pi == "control" | object$include_pi == "cox") {
  #   x_pcox = cbind(x_pcox, pihat_pred)
  # }

  cat("Starting Prediction \n")
  n_chains = length(object$coda_chains)

  do_type_config <- .get_do_type(ncores)
  `%doType%` <- do_type_config$doType

  chain_out <- foreach::foreach(iChain = 1:n_chains) %doType% {
    tree_files = .get_chain_tree_files(save_tree_directory, iChain, arg1 = flag, arg2 = irep)
    cat("Starting to Predict Chain ", iChain, "\n")
    # Note: what obtained below are prediction made from the saved ndraw posterior draws of trees
    modts = TreeSamples$new()
    modts$load(tree_files$mod_trees)
    Tm = modts$predict(t(x_pm)) #pure mod tree fit, dim(ndraws,n)

    conts = TreeSamples$new()
    conts$load(tree_files$con_trees)
    Tc = conts$predict(t(x_pc)) # pure con tree fit
    
    # temp_con1 = 
    # 
    # x_pcox = cbind(x_pcox, )
    # coxts = TreeSamples$new()
    # coxts$load(tree_files$coxcon_trees)
    # Tcoxcon = coxts$predict(t(x_pcox))

    list(Tm = Tm, Tc = Tc)#, Tcoxcon = Tcoxcon)
  }

  all_pred_m1 = c()
  all_pred_m0 = c()
  #all_mu = c()
  all_tau_m = c()
  
  #all_cure = c()

  chain_list = list()

  for (iChain in 1:n_chains) {
    Tm = chain_out[[iChain]]$Tm
    Tc = chain_out[[iChain]]$Tc
    #Tcure = chain_out[[iChain]]$Tcure

    this_chain_bcf_out = object$raw_chains[[iChain]]

    #null_inter = object$muy
    b1 = this_chain_bcf_out$b1
    b0 = this_chain_bcf_out$b0
    #mu_scale = this_chain_bcf_out$mu_scale

    # get tau, mu, and y: the three parts that we care about in prediction
    p_m = object$dim_m
    nobs = dim(x_pc)[1]#dim(object$mhat)[2]
    
    confit <- list()
    for (i in 1:p_m) {
      mat = cbind(Tc[, i])
      for (j in 1:(nobs-1)) {
        mat = cbind(mat, Tc[, j*p_m+i])
      }
      confit[[i]] = mat
    }
    
    modfit <- list()
    tau_post <- list()
    for (i in 1:p_m) {
      mat = cbind(Tm[, i])
      for (j in 1:(nobs-1)) {
        mat = cbind(mat, Tm[, j*(p_m+1)+i]) # or, dim_m +1
      }
      modfit[[i]] = mat
    }
    
    pred_m1 = array(NA, dim = c(dim(object$mhat)[1], nobs, p_m))
    pred_m0 = array(NA, dim = c(dim(object$mhat)[1], nobs, p_m))
    for (i in 1:p_m) {
      pred_m1[,,i] = confit[[i]] + modfit[[i]]*b1
      pred_m0[,,i] = confit[[i]] + modfit[[i]]*b0 
    }
    
    tau_m = array(NA, dim = c(dim(object$mhat)[1], nobs, p_m))
    for (i in 1:p_m) {
      tau_m[,,i] = (b1-b0)*modfit[[i]]
    }
    #mu = T_c + Tm*b0 #null_inter + Tc*mu_scale + Tm*b0
    #tau = (b1 - b0)*Tm #mu+tau = mu_trt
    #yhat = mu + t(t(tau)*a_pred)

    #all_yhat = rbind(all_yhat, yhat)
    #all_mu = rbind(all_mu, mu)
    all_pred_m1 = abind(all_pred_m1, pred_m1, along = 1)
    all_pred_m0 = abind(all_pred_m0, pred_m0, along = 1)
    all_tau_m = abind(all_tau_m, tau_m, along = 1)
    
    scalar_df <- data.frame("tau_m_bar" = apply(tau_m, c(1,3), mean),
                            #"mu_bar" = matrixStats::rowWeightedMeans(mu, w = NULL),
                            "pred_m1_bar" = apply(pred_m1, c(1,3), mean),
                            "pred_m0_bar" = apply(pred_m0, c(1,3), mean)
                            )

    chain_list[[iChain]] <- coda::as.mcmc(scalar_df)
  }

  .cleanup_after_par(do_type_config)

  list(tau_m = all_tau_m,
       #mu = all_mu,
       #yhat = all_yhat,
       pred_m1 = all_pred_m1,
       pred_m0 = all_pred_m0,
       #uncured_prob = all_cure,
       coda_chains = coda::as.mcmc.list(chain_list))
}


# TO BE Modified
#' @export Ypredict.bthm
#' @export
Ypredict.bthm <- function(object, x_pred_coxcon, x_pred_moderate, m_pred, #x_pred_coxcon, 
                          pihat_pred, #a_pred,
                          save_tree_directory, flag = 1, irep = 1, ncores = 1, 
                          ...) {
  if(any(is.na(x_pred_coxcon))) stop("Missing values in x_pred_coxcon")
  if(any(is.na(x_pred_moderate))) stop("Missing values in x_pred_moderate")
  #if(any(is.na(x_pred_coxcon))) stop("Missing values in x_pred_coxcon")
  if(any(is.na(pihat_pred))) stop("Missing values in pihat_pred")
  if(any(is.na(m_pred))) stop("Missing values in m_pred")
  
  pihat_pred = as.matrix(pihat_pred)
  
  x_pcox = matrix(x_pred_coxcon, ncol = ncol(x_pred_coxcon))
  x_pm = matrix(x_pred_moderate, ncol = ncol(x_pred_moderate))
  #x_pcox = matrix(x_pred_cure, ncol = ncol(x_pred_coxcon))
  m = matrix(m_pred, ncol = ncol(m_pred))
  
  if(object$include_pi == "both" | object$include_pi == "cox" | object$include_pi == "control") {
    x_pcox = cbind(x_pcox, pihat_pred)
  }
  
  if(object$include_pi == "both" | object$include_pi == "moderate") {
    x_pm = cbind(x_pm, pihat_pred)
  }
  
  if (nrow(x_pcox) == nrow(m)) {
    x_pcox = cbind(x_pcox, m)
  } else { cat("error: x_pcox and m_pred dim mismatch.\n")}
  
  # if(object$include_pi == "both" | object$include_pi == "control" | object$include_pi == "cox") {
  #   x_pcox = cbind(x_pcox, pihat_pred)
  # }
  
  cat("Starting Prediction \n")
  n_chains = length(object$coda_chains)
  
  do_type_config <- .get_do_type(ncores)
  `%doType%` <- do_type_config$doType
  
  chain_out <- foreach::foreach(iChain = 1:n_chains) %doType% {
    tree_files = .get_chain_tree_files(save_tree_directory, iChain, arg1 = flag, arg2 = irep)
    cat("Starting to Predict Chain ", iChain, "\n")
    # Note: what obtained below are prediction made from the saved ndraw posterior draws of trees
    modts = TreeSamples$new()
    modts$load(tree_files$mod_trees)
    Tm = modts$predict(t(x_pm)) #pure mod tree fit, dim(ndraws,n)
    
    coxts = TreeSamples$new()
    coxts$load(tree_files$coxcon_trees)
    Tcox = coxts$predict(t(x_pcox)) # pure con tree fit
    
    # temp_con1 = 
    # 
    # x_pcox = cbind(x_pcox, )
    # coxts = TreeSamples$new()
    # coxts$load(tree_files$coxcon_trees)
    # Tcoxcon = coxts$predict(t(x_pcox))
    
    list(Tm = Tm, Tcox = Tcox)#, Tcoxcon = Tcoxcon)
  }
  
  all_pred_y1 = c()
  all_pred_y0 = c()
  #all_mu = c()
  all_tau = c()
  
  #all_cure = c()
  
  chain_list = list()
  
  for (iChain in 1:n_chains) {
    Tm = chain_out[[iChain]]$Tm
    Tcox = chain_out[[iChain]]$Tcox
    #Tcure = chain_out[[iChain]]$Tcure
    
    this_chain_bcf_out = object$raw_chains[[iChain]]
    
    # get tau, mu, and y: the three parts that we care about in prediction
    p_m = object$dim_m
    nobs = dim(x_pcox)[1]#dim(object$mhat)[2]
    
    
    modmat = cbind(Tm[, p_m+1])
    for (j in 1:(nobs-1)) {
      modmat = cbind(modmat, Tm[, (j+1)*(p_m+1)]) # or, dim_m +1
    }
    
    
    #pred_y1 = array(NA, dim = c(dim(object$yhat)[1], nobs))
    #pred_y0 = array(NA, dim = c(dim(object$yhat)[1], nobs))
    pred_y1 = Tcox + modmat
    pred_y0 = Tcox
    
    
    #tau_m = array(NA, dim = c(dim(object$mhat)[1], nobs, p_m))
    #for (i in 1:p_m) {
    #  tau_m[,,i] = (b1-b0)*modfit[[i]]
    #}
    #mu = T_c + Tm*b0 #null_inter + Tc*mu_scale + Tm*b0
    #tau = (b1 - b0)*Tm #mu+tau = mu_trt
    #yhat = mu + t(t(tau)*a_pred)
    
    #all_yhat = rbind(all_yhat, yhat)
    #all_mu = rbind(all_mu, mu)
    all_pred_y1 = rbind(all_pred_y1, pred_y1)
    all_pred_y0 = rbind(all_pred_y0, pred_y0)
    all_tau = rbind(all_tau, modmat)
    
    scalar_df <- data.frame("tau_hzd_bar" = matrixStats::rowWeightedMeans(modmat, w = NULL),
                            #"mu_bar" = matrixStats::rowWeightedMeans(mu, w = NULL),
                            "pred_y1_bar" = matrixStats::rowWeightedMeans(pred_y1, w = NULL),
                            "pred_y0_bar" = matrixStats::rowWeightedMeans(pred_y0, w = NULL)
    )
    
    chain_list[[iChain]] <- coda::as.mcmc(scalar_df)
  }
  
  .cleanup_after_par(do_type_config)
  
  list(tau_hzd = all_tau,
       #mu = all_mu,
       #yhat = all_yhat,
       pred_y1 = all_pred_y1,
       pred_y0 = all_pred_y0,
       #uncured_prob = all_cure,
       coda_chains = coda::as.mcmc.list(chain_list))
}





# TO BE Modified
#' @export Ypostpred.bthm
#' @export
Ypostpred.bthm <- function(object, x_pred_coxcon, x_pred_moderate, m_post, #x_pred_coxcon, 
                          pihat_pred, ndraws, #a_pred, 
                          save_tree_directory, flag = 1, irep = 1, ncores = 1,
                          ...) {
  if(any(is.na(x_pred_coxcon))) stop("Missing values in x_pred_coxcon")
  if(any(is.na(x_pred_moderate))) stop("Missing values in x_pred_moderate")
  #if(any(is.na(x_pred_coxcon))) stop("Missing values in x_pred_coxcon")
  if(any(is.na(pihat_pred))) stop("Missing values in pihat_pred")
  if(any(is.na(m_post))) stop("Missing values in m_post")
  
  pihat_pred = as.matrix(pihat_pred)
  
  x_pcox = matrix(x_pred_coxcon, ncol = ncol(x_pred_coxcon))
  x_pm = matrix(x_pred_moderate, ncol = ncol(x_pred_moderate))
  #x_pcox = matrix(x_pred_cure, ncol = ncol(x_pred_coxcon))
  # If we are changing it to predict for the m[,,i] with the ith iteration/posterior draw of the tree, then 
  # the dim of m should be n_predsample x p_m x n_postpred_draws
  if(dim(m_post)[1] != nrow(x_pred_coxcon)) stop("unmatched sample size for x and m")
  if(dim(m_post)[3]/ndraws != dim(object$coda_chains[[1]])[1]) stop("unmatched number of iteration/postpred draws for m_post and the fitted object")
  
  if(object$include_pi == "both" | object$include_pi == "cox" | object$include_pi == "control") {
    x_pcox = cbind(x_pcox, pihat_pred)
  }
  
  if(object$include_pi == "both" | object$include_pi == "moderate") {
    x_pm = cbind(x_pm, pihat_pred)
  }
  
  # if(object$include_pi == "both" | object$include_pi == "control" | object$include_pi == "cox") {
  #   x_pcox = cbind(x_pcox, pihat_pred)
  # }
  
  cat("Starting Prediction \n")
  n_chains = length(object$coda_chains)
  
  do_type_config <- .get_do_type(ncores)
  `%doType%` <- do_type_config$doType
  
  chain_out <- foreach::foreach(iChain = 1:n_chains) %doType% {
    tree_files = .get_chain_tree_files(save_tree_directory, iChain, arg1 = flag, arg2 = irep)
    cat("Starting to Predict Chain ", iChain, "\n")
    # Note: what obtained below are prediction made from the saved ndraw posterior draws of trees
    modts = TreeSamples$new()
    modts$load(tree_files$mod_trees)
    Tm = c()
    #Tm = modts$predict(t(x_pm)) #pure mod tree fit, dim(ndraws,n)
    
    coxts = TreeSamples$new()
    coxts$load(tree_files$coxcon_trees)
    Tcox = c()
    for (itr in 1:dim(object$coda_chains[[1]])[1]) {
      for (predrep in 1:ndraws) {
        if (nrow(x_pcox) == nrow(array(m_post[,,(itr-1)*ndraws + predrep], dim = dim(m_post)[1:2]))) {
          x_pcox_pp = cbind(x_pcox, array(m_post[,,(itr-1)*ndraws + predrep], dim = dim(m_post)[1:2]))
      } else { cat(paste("error: x_pcox and m_postpred[,,", predrep,"] dim mismatch.\n"))}
      Tcox = rbind(Tcox, coxts$predict_i(t(x_pcox_pp), itr-1)) # so this Tcox will be npredrep*n?
      Tm = rbind(Tm, modts$predict_i(t(x_pm), itr-1))
      }
    }
     # pure con tree fit
    
    # temp_con1 = 
    # 
    # x_pcox = cbind(x_pcox, )
    # coxts = TreeSamples$new()
    # coxts$load(tree_files$coxcon_trees)
    # Tcoxcon = coxts$predict(t(x_pcox))
    
    list(Tm = Tm, Tcox = Tcox)#, Tcoxcon = Tcoxcon)
  }
  
  all_pred_y1 = c()
  all_pred_y0 = c()
  #all_mu = c()
  all_tau = c()
  
  #all_cure = c()
  
  chain_list = list()
  
  for (iChain in 1:n_chains) {
    Tm = chain_out[[iChain]]$Tm
    Tcox = chain_out[[iChain]]$Tcox
    #Tcure = chain_out[[iChain]]$Tcure
    
    this_chain_bcf_out = object$raw_chains[[iChain]]
    
    # get tau, mu, and y: the three parts that we care about in prediction
    p_m = object$dim_m
    nobs = dim(x_pcox)[1]#dim(object$mhat)[2]
    
    
    modmat = cbind(Tm[, p_m+1])
    for (j in 1:(nobs-1)) {
      modmat = cbind(modmat, Tm[, (j+1)*(p_m+1)]) # or, dim_m +1
    }
    
    
    #pred_y1 = array(NA, dim = c(dim(object$yhat)[1], nobs))
    #pred_y0 = array(NA, dim = c(dim(object$yhat)[1], nobs))
    pred_y1 = Tcox + modmat
    pred_y0 = Tcox
    
    
    #tau_m = array(NA, dim = c(dim(object$mhat)[1], nobs, p_m))
    #for (i in 1:p_m) {
    #  tau_m[,,i] = (b1-b0)*modfit[[i]]
    #}
    #mu = T_c + Tm*b0 #null_inter + Tc*mu_scale + Tm*b0
    #tau = (b1 - b0)*Tm #mu+tau = mu_trt
    #yhat = mu + t(t(tau)*a_pred)
    
    #all_yhat = rbind(all_yhat, yhat)
    #all_mu = rbind(all_mu, mu)
    all_pred_y1 = rbind(all_pred_y1, pred_y1)
    all_pred_y0 = rbind(all_pred_y0, pred_y0)
    
    pred_rddt = matrix(rep(object$rddt, ncol(all_pred_y1)*ndraws), nrow = dim(object$coda_chains[[1]])[1]*ndraws)
    #all_pred_y1 = all_pred_y1 - pred_rddt
    #all_pred_y0 = all_pred_y0 - pred_rddt
    
    all_tau = rbind(all_tau, modmat)
    
    scalar_df <- data.frame("tau_hzd_bar" = matrixStats::rowWeightedMeans(modmat, w = NULL),
                            #"mu_bar" = matrixStats::rowWeightedMeans(mu, w = NULL),
                            "pred_y1_bar" = matrixStats::rowWeightedMeans(pred_y1, w = NULL),
                            "pred_y0_bar" = matrixStats::rowWeightedMeans(pred_y0, w = NULL)
    )
    
    chain_list[[iChain]] <- coda::as.mcmc(scalar_df)
  }
  
  .cleanup_after_par(do_type_config)
  
  list(tau_hzd = all_tau,
       #mu = all_mu,
       #yhat = all_yhat,
       pred_y1 = all_pred_y1,
       pred_y0 = all_pred_y0,
       #uncured_prob = all_cure,
       coda_chains = coda::as.mcmc.list(chain_list))
}



