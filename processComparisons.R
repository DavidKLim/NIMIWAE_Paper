reverse_norm_MIWAE = function(x,norm_means,norm_sds){
  xnew=matrix(nrow=nrow(x),ncol=ncol(x))
  for(i in 1:ncol(xnew)){
    xnew[,i]=(x[,i]*(norm_sds[i]))+norm_means[i]
  }
  return(xnew)
}

NRMSE = function(x,xhat,Missing){
  x=as.matrix(x);xhat=as.matrix(xhat);Missing=as.matrix(Missing)
  #x = (x-colMeans(x))/apply(x,2,sd)
  #xhat = (xhat-colMeans(x))/apply(x,2,sd)
  # Missing=1 --> observed

  MSE=rep(NA,ncol(x))
  RMSE=rep(NA,ncol(x))
  NRMSE=rep(NA,ncol(x))
  for(j in 1:ncol(x)){
    if(all(Missing[,j]==1)){next}
    norm_term = (max(x[Missing[,j]==0,j])-min(x[Missing[,j]==0,j])) # in case denom is 0
    # norm_term = (max(x[,j])-min(x[,j]))
    # norm_term = sd(x[,j])
    # norm_term = sd(x[Missing[,j]==0,j])
    MSE[j] = mean((x[Missing[,j]==0,j]-xhat[Missing[,j]==0,j])^2)
    RMSE[j] = sqrt(MSE[j])
    NRMSE[j] = RMSE[j]/norm_term
  }
  MSE=mean(MSE,na.rm=T); RMSE=mean(RMSE,na.rm=T); NRMSE=mean(NRMSE,na.rm=T)

  # MSE = mean((x[Missing==0]-xhat[Missing==0])^2)
  # RMSE = sqrt(MSE)
  # NRMSE = RMSE / sd(x[Missing==0])
  L1 = mean(abs(x[Missing==0]-xhat[Missing==0]))
  L2 = mean((x[Missing==0]-xhat[Missing==0])^2)
  return(list(MSE=MSE,RMSE=RMSE,NRMSE=NRMSE,L1=L1,L2=L2))
}

overlap_hists=function(x1,x2,x3=NULL,lab1="Truth",lab2="Imputed",lab3="...",
                       title="MNAR Missing Values, Truth vs Imputed, Missing column"){
  library(ggplot2)
  x1=data.frame(value=x1); x1$status=lab1
  x2=data.frame(value=x2); x2$status=lab2
  if(!is.null(x3)){x3=data.frame(value=x3); x3$status=lab3; df=rbind(x1,x2,x3)
  }else{df = rbind(x1,x2)}
  p = ggplot(df,aes(value,fill=status)) + geom_density(alpha=0.2, adjust=1/5) + ggtitle(title) + xlim(quantile(df$value,c(0.01,0.99),na.rm=T))
  print(p)
}

output_file.name=function(dir_name,method=c("IMIWAE_unsup","NIMIWAE_unsup","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE_unsup"),
                          mechanism,miss_pct,arch,rdeponz, init){
  if(grepl("^NIMIWAE", method)){
    yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    if(init=="default"){NIM_pref = ""} else if(init=="alt"){NIM_pref="alt_init/"}
    file.name=sprintf("%s/%sres_NIMIWAE_%s_%d_%s_%s",
                      dir_name,NIM_pref,method,mechanism,miss_pct,arch,yesrdeponz)
  }else if(grepl("^IMIWAE",method)){
    yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    file.name=sprintf("%s/Ignorable/res_NIMIWAE_%s_%d_%s_%s",
                      dir_name,method,mechanism,miss_pct,arch,yesrdeponz)
  }else{
    file.name=sprintf("%s/res_%s_%s_%d",
                      dir_name,method,mechanism,miss_pct) # for miwae, default = Normal (StudentT can be done later)
  }
  print(file.name)
  file.name=sprintf("%s.RData",file.name)
  return(file.name)
}

process_results=function(data.file.name, file.name, method=c("IMIWAE_unsup","NIMIWAE_unsup","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE_unsup")){
  call_name=match.call()

  # load data and split into training/valid/test sets
  load(data.file.name)

  ## turn this back on (1/3)
  if(is.null(g)){  # turned saving g back on, but for prev results (seed locked)
    ratios=c(train = .8, valid = .2)
    set.seed(333)
    g = sample(cut(
      seq(nrow(data)),
      nrow(data)*cumsum(c(0,ratios)),
      labels = names(ratios)
    ))
  }

  classes = fit_data$classes
  P=ncol(data); N=nrow(data)
  if(grepl("_sup",method)){
    ## supervised methods
    # data = cbind(data, classes)
    data = cbind(data, as.numeric(as.factor(classes)))

    Missing = cbind(Missing, rep(1, nrow(Missing)))  # include 1's for classes (fully observed)
  }

  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  if(!is.null(classes)){ classess = split(classes,g) }else{classess = list()}

  ## turn this back on (2/3)
  if(is.null(datas$test)){
    datas$test = datas$train
    Missings$test = Missings$train
    classess$test = classess$train
  }

  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  # MIWAE and NIMIWAE only
  load(file.name)
  print(file.name)
  # fit=eval(parse(text=paste("res",method,sep="_")))

  results2 = list()  # for imputation
  #xhat=reverse_norm_MIWAE(fit$xhat,norm_means,norm_sds)   # already reversed
  if(grepl("NIMIWAE",method) | grepl("^IMIWAE",method)){
    xhat=res_NIMIWAE$xhat
    time = res_NIMIWAE$time
  }else if(grepl("MIWAE",method)){
    xhat=res_MIWAE$xhat_rev
    time = res_MIWAE$time
  }else if(grepl("HIVAE",method)){
    xhat=res_HIVAE$data_reconstructed
    time = res_HIVAE$time
  }else if(grepl("VAEAC",method)){
    xhat_all = res_VAEAC$result    # this method reverses normalization intrinsically
    # average imputations
    xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
    n_imputations = res_VAEAC$train_params$n_imputations
    for(i in 1:nrow(datas$test)){
      xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
    }
    time = res_VAEAC$time
  }else if(grepl("MEAN",method)){
    # xhat = fit$xhat_rev
    if(is.null(res_MEAN$xhat_rev)){res_MEAN$xhat_rev = reverse_norm_MIWAE(res_MEAN$xhat_mean,norm_means,norm_sds)}
    xhat = res_MEAN$xhat_rev
    time = res_MEAN$time
  }else if(grepl("MF",method)){
    # xhat = fit$xhat_rev
    if(is.null(res_MF$xhat_rev)){res_MF$xhat_rev = reverse_norm_MIWAE(res_MF$xhat_mf,norm_means,norm_sds)}
    xhat = res_MF$xhat_rev
    time = res_MF$time
  }else if(grepl("MICE",method)){

    xhat=res_MICE$xhat
    time = res_MICE$time
  }

  ## turn this back on (3/3)
  if(all(dim(xhat) != dim(datas$test))){print("dim(xhat):"); print(dim(xhat)); print("dim(test data):"); print(dim(datas$test)); stop("Dimensions wrong")}

  # check same xhat:
  print("Mean Squared Error (Observed): should be 0")
  print(mean((xhat[Missings$test==1] - datas$test[Missings$test==1])^2))    # should be 0
  print("Mean Squared Error (Missing):")
  print(mean((xhat[Missings$test==0] - datas$test[Missings$test==0])^2))

  # Imputation metrics

  imputation_metrics=NRMSE(x=datas$test, xhat=xhat, Missing=Missings$test)
  #imputation_metrics=NRMSE(x=xfull, xhat=xhat, Missing=Missings$test)

  # Other metrics (names aren't consistent)
  #LB=fit$LB; time=fit$time


  results = c(unlist(imputation_metrics))

  # ratio = 0.8; n_train = floor(ratio*nrow(xhat)); n_test = nrow(xhat) - n_train
  # idx = c( rep(T, n_train) , rep(F, n_test) )
  #
  # fit_pred_imputed = predict_classes(X_train=as.matrix(xhat[idx,]), y_train=classess$test[idx], X_test=as.matrix(xhat[!idx,]), y_test=classess$test[!idx])
  # fit_pred_true = predict_classes(X_train=as.matrix(datas$test[idx,]), y_train=classess$test[idx], X_test=as.matrix(datas$test[!idx,]), y_test=classess$test[!idx])
  #
  # fits_pred = list(imputed = fit_pred_imputed,
  #                  true = fit_pred_true)
  # fit$fits_pred = fits_pred

  return(list(results=results, call=call_name, time = time))
}
process_results_MI = function(data.file.name, file.name, method, dir_name=""){
  call_name=match.call()

  # load data and split into training/valid/test sets
  load(data.file.name)


  ## since g=NULL is being saved rn. (need to change)
  if(is.null(g)){
    ratios=c(train = .8, valid = .2)
    set.seed(333)
    g = sample(cut(
      seq(nrow(data)),
      nrow(data)*cumsum(c(0,ratios)),
      labels = names(ratios)
    ))
  }

  classes = fit_data$classes

  P=ncol(data); N=nrow(data)

  # if(grepl("_sup",method)){
  #   ## supervised methods
  #   data = cbind(data, classes)
  #   Missing = cbind(Missing, rep(1, nrow(Missing)))  # include 1's for classes (fully observed)
  # }

  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  classess = split(classes,g)

  datas$test = datas$train
  Missings$test = Missings$train
  classess$test = classess$train

  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  # MIWAE and NIMIWAE only
  load(file.name)
  print(file.name)
  # fit=eval(parse(text=paste("res",method,sep="_")))

  # if(!(grepl("NIMIWAE",method) | grepl("MICE",method) )){stop("Only MICE or NIMIWAE for MI")}

  results2 = list()  # for imputation
  #xhat=reverse_norm_MIWAE(fit$xhat,norm_means,norm_sds)   # already reversed
  if(grepl("NIMIWAE",method)){
    xhat=res_NIMIWAE$xhat
  }else if(grepl("MIWAE",method)){
    xhat=res_MIWAE$xhat_rev
  }else if(grepl("HIVAE",method)){
    xhat=res_HIVAE$data_reconstructed
  }else if(grepl("VAEAC",method)){
    xhat_all = res_VAEAC$result    # this method reverses normalization intrinsically
    # average imputations
    xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
    n_imputations = res_VAEAC$train_params$n_imputations
    for(i in 1:nrow(datas$test)){
      xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
    }
  }else if(grepl("MEAN",method)){
    # xhat = fit$xhat_rev
    # if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mean,norm_means,norm_sds)}
    xhat = res_MEAN$xhat_rev
  }else if(grepl("MF",method)){
    # xhat = fit$xhat_rev
    # if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mf,norm_means,norm_sds)}
    xhat = res_MF$xhat_rev
  }else if(grepl("MICE",method)){

    xhat=res_MICE$xhat

  }

  X = datas$test[,1:P]; mask = Missings$test[,1:P]   ## take out response (binary/cat) variable

  if(all(dim(xhat) != dim(datas$test))){print("dim(xhat):"); print(dim(xhat)); print("dim(test data):"); print(dim(datas$test)); stop("Dimensions wrong")}

  ### MI + Regression

  m=50
  library(mice)
  library(pROC)
  if(grepl("NIMIWAE", method)){
    res = res_NIMIWAE; rm(res_NIMIWAE)

    norm_means = res$norm_means[1:P]; norm_sds = res$norm_sds[1:P]
    library("data.table"); library("readr"); library("rhdf5")
    # imp_weights = as.matrix(fread(sprintf("%s/IWs.csv",dir_name)))    # m(n_imputations) x N
    # miss_XY = as.matrix(fread(sprintf("%s/miss_XY.csv",dir_name)))    # for just missing entries saved (indices also saved in this separate file)
    samples = rhdf5::H5Fopen(sprintf("%s/samples.h5",dir_name))  # samples has $IWs $miss_XY and $Xm0 to $Xm#
    imp_weights = t(samples$IWs)    # for some reason, rhdf5 always transposes what's saved in python
    miss_XY = t(samples$miss_XY)
    # rhdf5::h5closeAll()  # closing here sometimes causes issues..

    N1 = res$opt_params$L * res$opt_params$M  # number of initial samples (should be # samples of Z (L) * # samples of Xm (M) in testing)
    N2 = m   # number of importance-weighted resamples. this can be different from m (could be like 2*m too)

    fname = sprintf("%s/SIR_samp%d.csv",dir_name,N2)  # check to see if all SIR samples were already drawn. skip if they've already been saved
    fname1 = sprintf("%s/NIMIWAE_MI_models.RData",dir_name)

    if(!file.exists(fname) & !file.exists(fname1)){

      fname0 = sprintf("%s/multinom_draws.csv",dir_name)
      if(!file.exists(fname0)){
        # t0 = Sys.time()
        draws = matrix(nrow=nrow(X), ncol=N2)
        # set.seed(1)
        for(i in 1:ncol(imp_weights)){  ## for each observation
          draws[i,] = apply(rmultinom(N2, 1, imp_weights[,i]), 2, which.max)   # sample N2 times --> which imputed datasets to choose for that sample
        }
        # print(as.numeric(Sys.time() - t0, units="secs"))
        fwrite(draws, file=fname0)
      } else{ draws = as.matrix(fread(fname0))}
    }

    if(!file.exists(fname) & !file.exists(fname1)){

      ### loop of N1 --> filling in each N2 SIR samples --> speedup of 7-8 times
      samps=list()

      ts = Sys.time()
      for(j in 1:N2){
        samps[[j]] = datas$test   # initialize
      }
      unique_draws = unique(c(draws))
      for(j in 1:N1){
        print(j)
        if(!(j %in% unique_draws)){ next }
        id = which(draws == j,arr.ind=T)    # row: observation index, col: SIR sample index. current Xm j --> row goes in for SIR sample col
        unique_SIRs = unique(id[,2])     # unique SIR's that we need to replace from this drawn Xm[j]
        for(k in 1:length(unique_SIRs)){
          id_rows = id[id[,2]==unique_SIRs[k],1]   # rows of data matrix that needs to be replaced for SIR sample k, pertaining to this Xm
          # samps[[k]][id_rows,][mask[id_rows,]==0] = Xms[[j]][((miss_XY[,1]+1) %in% id_rows)]
          # samps[[k]][id_rows,][mask[id_rows,]==0] = t( eval(parse(text = sprintf("samples$Xm%d",j-1))) )[((miss_XY[,1]+1) %in% id_rows)]
          # t(samps[[k]][id_rows,])[t(mask[id_rows,])==0] = t( eval(parse(text = sprintf("samples$Xm%d",j-1))) )[((miss_XY[,1]+1) %in% id_rows)]
          ## can't replace transpose of matrix --> make dummy matrix that is transpose we want to replace, replace, then untranspose.
          dummy = t(samps[[k]][id_rows,])
          dummy[t(mask[id_rows,])==0] = t( eval(parse(text = sprintf("samples$Xm%d",j-1))) )[((miss_XY[,1]+1) %in% id_rows)]
          samps[[k]][id_rows,] = t(dummy)
        }
      }
      t2 = as.numeric(Sys.time()-ts, units="secs")
      for(j in 1:N2){
        fname0 = sprintf("%s/SIR_samp%d.csv",dir_name,j)
        fwrite(samps[[j]], file=fname0)
      }
    } else{
      samps = list()
      for(j in 1:N2){
        fname0 = sprintf("%s/SIR_samp%d.csv",dir_name,j)
        samps[[j]] = as.matrix(fread(fname0))
      }
    }

    ######  fname1: models seem to be too large to load on 32GB ram...
    # if(!file.exists(fname1)){
    reg_fit = list(); AUCs=rep(NA,m)
    for(i in 1:m){
      print(paste("Fitting model", i, "of", m))
      dat = data.frame(cbind(samps[[i]], y=classess$test))   # append back the response variable
      reg_fit[[i]] = glm(as.factor(y) ~ 1 + ., data=dat, family=binomial(link="logit"))
      # if(intercept){ reg_fit[[i]] = glm(as.factor(y) ~ 1 + ., data=dat, family=binomial(link="logit"))   # check imps[[1]] for response variable name in dat
      # }else{ reg_fit[[i]] = glm(as.factor(y) ~ 0 + ., data=dat, family=binomial(link="logit")) }   # check imps[[1]] for response variable name in dat

      samps[[i]] = NA; rm(dat)   # remove these memories
      gc()
    }
    # save(list=c("reg_fit","AUCs"),file=fname1)
    # } else{
    #   gc(); load(fname1)
    # }
  }else if(grepl("MICE", method)){

    reg_fit = list()
    AUCs=rep(NA,m)
    for(i in 1:res_MICE$m){
      ## Easy way to use full model, response is last variable in each completed dataset
      ## Include an intercept
      print(paste("Fitting model", i, "of", m))
      if(grepl("_sup",method)){
        dat = complete(res_MICE,i); colnames(dat)[ncol(dat)] = "y"
      } else{
        dat = data.frame(cbind(complete(res_MICE,i), y=classess$test))
      }

      reg_fit[[i]] = glm(as.factor(y) ~ 1 + ., data=dat, family=binomial(link="logit"))
      # if(intercept){ reg_fit[[i]] = glm(as.factor(y) ~ 1 + ., data=dat, family=binomial(link="logit"))
      # }else{ reg_fit[[i]] = glm(as.factor(y) ~ 0 + ., data=dat, family=binomial(link="logit")) }

      rm(dat); gc()

      ## No intercept
      # fit[[i]] = glm(update(imp$formulas[length(imp$formulas)][[1]], as.factor(.) ~ .),
      #                data=complete(imp,i), family=binomial(link="logit"))
    }
  }else{
    dat = data.frame(cbind(xhat,classess$test)); colnames(dat)[ncol(dat)] = "y"
    reg_fit = glm(as.factor(y) ~ 1 + ., data = dat, family=binomial(link="logit"))
    # reg_fit = if(intercept){ glm(as.factor(y) ~ 1 + ., data = dat, family=binomial(link="logit"))
    # } else{ glm(as.factor(y) ~ 0 + ., data = dat, family=binomial(link="logit")) }

    AUCs=NA
  }

  if(grepl("NIMIWAE", method) | grepl("MICE",method)){
    tab <- summary(pool(reg_fit), "all", conf.int = TRUE)
    itab = tab[, c("estimate", "std.error", "2.5 %", "97.5 %")]
    tab = tab[-1, c("estimate", "std.error", "2.5 %", "97.5 %")]
    # if(!intercept){ tab = tab[-1, c("estimate", "std.error", "2.5 %", "97.5 %")]
    # } else{ tab = tab[, c("estimate", "std.error", "2.5 %", "97.5 %")] }
  } else{
    tab <- summary(reg_fit)$coef
    CIs = confint(reg_fit)
    tab = cbind(tab[,1:2],CIs)
    colnames(tab) = c("estimate", "std.error","2.5 %", "97.5 %")
    itab = tab
    colnames(itab) = c("estimate", "std.error","2.5 %", "97.5 %")
    tab = tab[-1,]
    # if(!intercept){ tab = tab[-1,] }
  }

  print(tab)
  # print(AUCs)
  print("Finished process_MI")
  rhdf5::h5closeAll()

  dat = cbind(datas$train, classess$train)
  colnames(dat)[ncol(dat)]="y"
  fit = glm(as.factor(y) ~ 1 + ., data=dat, family=binomial(link="logit"))
  trues = fit$coef[-1]
  itrues = fit$coef
  print("Finished process_MI fx")
  return(list(tab=tab, itab=itab, reg_fit=reg_fit, fit_data=fit_data, names=colnames(data), trues = trues, itrues=itrues))#, AUCs=AUCs))

}

processComparisons=function(dir_name="Results/SIM1/phi5",mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(15,25,35),
                            methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),
                            imputation_metric=c("MSE","NRMSE","L1","L2"), arch=NULL, rdeponz=NULL, outfile=NULL, init="default"){

  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(reshape2)

  if(init=="default"){NIM_pref = ""}else if(init=="alt"){NIM_pref = "/alt_init"}
  dir_name2 = sprintf("%s%s", dir_name, NIM_pref)
  ifelse(!dir.exists(dir_name2), dir.create(dir_name2,recursive=T), F)

  g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
  mats_res = list(); mats_params=list()
  print("Compiling results...")
  list_res = list()
  times=list()
  for(ii in 1:length(miss_pct)){
    params=list()
    # input true probs, no learning R, Normal distrib, input_r="r", vary rdeponz and sample_r (4)
    index=1

    for(i in 1:length(mechanisms)){for(j in 1:length(methods)){
      data.file.name=sprintf("%s/data_%s_%d.RData",dir_name,mechanisms[i],miss_pct[ii])
      file.name = output_file.name(dir_name=dir_name,method=methods[j], mechanism=mechanisms[i],
                                   miss_pct=miss_pct[ii], arch=arch, rdeponz=rdeponz, init=init)
      list_res[[index]]=process_results(data.file.name,file.name,methods[j])
      params[[index]]=c(methods[j],mechanisms[i],miss_pct[ii])
      names(params[[index]])=c("method","mechanism","miss_pct")
      times[[index]] = list_res[[index]]$time
      index = index+1
    }}



    # flatten list to matrix
    mat_res = matrix(unlist(lapply(list_res,function(x)x$results)),ncol=length(list_res))
    rownames(mat_res)=names(list_res[[1]]$results)     # MSE, NRMSE, ...
    colnames(mat_res)=paste("case",c(1:length(list_res)),sep="")
    mat_res=t(mat_res)

    mat_params=matrix(unlist(params),ncol=length(params))
    rownames(mat_params)=c("method","mechanism","miss_pct")
    colnames(mat_params)=paste("case",c(1:ncol(mat_params)),sep="")
    mat_params=t(mat_params)

    df_bar = data.frame(cbind(mat_res,mat_params,rownames(mat_res)))
    for(c in 1:ncol(mat_res)){df_bar[,c]=as.numeric(as.character(df_bar[,c]))}
    colnames(df_bar)[ncol(df_bar)]="case"; df_bar$case = factor(df_bar$case,levels=paste("case",c(1:nrow(mat_params)),sep=""))
    df_bar$mechanism = factor(df_bar$mechanism,levels=c("MCAR","MAR","MNAR"))

    df_bar$method=as.character(df_bar$method)
    df_bar$method[df_bar$method=="MEAN"]="Mean"; df_bar$method[df_bar$method=="MF"]="MissForest"
    other_methods = methods[!grepl("NIMIWAE",methods)]; other_methods[other_methods=="MEAN"]="Mean"; other_methods[other_methods=="MF"]="MissForest"

    df_bar$method0 = df_bar$method
    NIMIWAE_methods = methods[grepl("NIMIWAE",methods)]; if(length(NIMIWAE_methods)==1){NIMIWAE_methods="NIMIWAE"; df_bar$method0[grepl("NIMIWAE",df_bar$method0)]="NIMIWAE"}
    df_bar$method0 = factor(df_bar$method0,
                            levels=c(other_methods[order(other_methods)], NIMIWAE_methods))
    # levels(df_bar$method)=methods[order(methods)]




    gg_color_hue <- function(n) {
      hues = seq(15, 375, length = n + 1)
      hcl(h = hues, l = 65, c = 100)[1:n]
    }
    colors = gg_color_hue(length(methods))

    p=ggplot(df_bar,aes(x=method0,y=eval(parse(text=imputation_metric)),fill=mechanism,color=mechanism))+
      geom_bar(stat="identity",position=position_dodge(.9),alpha=0.4)+#ylim(c(0,3))+#ylim(c(0,0.5))+
      labs(title=sprintf("%s Imputation Performance, %d%% Missing",imputation_metric, miss_pct[ii]),
           subtitle = "Imputation performance across missingness mechanisms",
           y = imputation_metric, x="Method")+
      theme(text=element_text(size = 20))

    if(is.null(outfile)){
      png(sprintf("%s/%s_competing_miss%d.png",dir_name2,imputation_metric,miss_pct[ii]),width=1200,height=500)
    } else{ png(outfile, width=1200, height=500)}
    #barplot(mat_res[rownames(mat_res)=="NRMSE",])
    print(p)
    dev.off()

    # save in mats_res
    mats_res[[ii]]=mat_res
    mats_params[[ii]]=mat_params
  }
  names(mats_res)=miss_pct
  return(list(res=mats_res, params=mats_params, times=times))
}

saveFigures = function(datasets=c("SIM"), sim_index=1:5, phi0=5, Ns=c(1e4, 1e5), Ps=c(25,100), Ds=c(2,8),
                       mechanisms=c("MCAR","MAR","MNAR"), miss_pct=c(15), methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),
                       outfile=NULL, init="default"){
  list_res=list(); index=1
  for(d in 1:length(datasets)){
  for(a in 1:length(Ns)){for(b in 1:length(Ps)){for(c in 1:length(Ds)){
    if(grepl("SIM",datasets[d])){ dataset=sprintf("%s_N%d_P%d_D%d", datasets[d], Ns[a], Ps[b], Ds[c])
    }else{ dataset = datasets[d] }

    for(s in 1:length(sim_index)){

        dir_name=sprintf("Results/%s/phi%d/sim%d",dataset,phi0,sim_index[s])
        data_dir_name=sprintf("Results/%s/phi%d/sim%d",dataset,phi0,sim_index[s])
        imputation_metrics=c("MSE","L1","L2","NRMSE","RMSE")
        for(i in 1:length(imputation_metrics)){
          res = processComparisons(dir_name,data_dir_name,mechanisms,miss_pct,methods,imputation_metrics[i],
                                        betaVAE=F, arch="IWAE", rdeponz=F, outfile=outfile, init=init)
          list_res[[index]] = res
          index = index+1
        }
      }
  }

    if(length(sim_index)>1){
      if(init=="default"){NIM_pref = ""}else if(init=="alt"){NIM_pref="/alt_init"}
      dir_name2 = sprintf("Results/%s/phi%d%s",dataset, phi0,NIM_pref)
      params=list_res[[1]]$params
      sims_list = list()
      for(i in 1:length(list_res)){ sims_list[[i]] = list_res[[i]]$res }

      dfs=data.frame()
      for(i in 1:length(sims_list)){
        df=data.frame(sims_list[[i]])
        df=cbind(df,params)
        names(df)=c("MSE","RMSE","NRMSE","L1","L2","method","mechanism","miss_pct")
        df$sim = sim_index[i]
        dfs = rbind(dfs,df)
      }

      bar_mean = aggregate(dfs[,1:length(list_res)], list(dfs$mechanism, dfs$method), mean)
      bar_sd = aggregate(dfs[,1:length(list_res)], list(dfs$mechanism, dfs$method), sd)
      bar_df = merge(bar_mean,bar_sd, by=c("Group.1","Group.2"))
      names(bar_df) = c("mechanism","method",
                        paste("mean_",c("MSE","RMSE","NRMSE","L1","L2"),sep=""),
                        paste("sd_",c("MSE","RMSE","NRMSE","L1","L2"),sep=""))


      library(ggplot2)
      gg_color_hue <- function(n) {
        hues = seq(15, 375, length = n + 1)
        hcl(h = hues, l = 65, c = 100)[1:n]
      }
      colors = gg_color_hue(length(methods))

      imputation_metric="L1"
      if(imputation_metric %in% c("L1","L2")){imputation_metric0=paste("Average ",imputation_metric,sep="")}

      bar_df$method = as.character(bar_df$method)

      bar_df$method[bar_df$method=="MEAN"]="Mean"; bar_df$method[bar_df$method=="MF"]="MissForest"
      # other_methods = methods[!grepl("NIMIWAE",methods)]; other_methods[other_methods=="MEAN"]="Mean"; other_methods[other_methods=="MF"]="MissForest"
      other_methods = methods[!grepl("IMIWAE",methods)]; other_methods[other_methods=="MEAN"]="Mean"; other_methods[other_methods=="MF"]="MissForest"
      # other_methods[other_methods=="IMIWAE_unsup"]="IMIWAE"; other_methods[other_methods=="MICE_unsup"]="MICE"
      other_methods[other_methods=="MICE_unsup"]="MICE"
      # NIMIWAE_methods = methods[grepl("NIMIWAE",methods)]
      NIMIWAE_methods = methods[grepl("IMIWAE",methods)]
      bar_df$method0 = bar_df$method
      # if(length(NIMIWAE_methods)==1){NIMIWAE_methods="NIMIWAE"; bar_df$method0[grepl("NIMIWAE",bar_df$method0)]="NIMIWAE"}
      NIMIWAE_methods[NIMIWAE_methods=="NIMIWAE_unsup"] = "NIMIWAE"; NIMIWAE_methods[NIMIWAE_methods=="IMIWAE_unsup"] = "IMIWAE"; bar_df$method0[bar_df$method0=="MICE_unsup"] = "MICE"
      bar_df$method0[bar_df$method0=="NIMIWAE_unsup"]="NIMIWAE"; bar_df$method0[bar_df$method0 == "IMIWAE_unsup"]="IMIWAE"
      bar_df$method0 = factor(bar_df$method0,
                              levels=c(other_methods[order(other_methods)],NIMIWAE_methods[order(NIMIWAE_methods)]))


      bar_df$method = bar_df$method0


      bar_df$mechanism = factor(bar_df$mechanism, levels=mechanisms)
      #eval(parse(text=paste("mean_",imputation_metric,sep="")))

      ##### Group by method, color by mechanism
      # p=ggplot(bar_df,aes(x=method0, y=eval(parse(text=paste("mean_",imputation_metric,sep=""))), fill=mechanism, color=mechanism))+
      #   geom_bar(stat="identity",position=position_dodge(.9),alpha=0.4,color="black")+#ylim(c(0,3))+#ylim(c(0,0.5))+
      #   geom_errorbar(aes(ymin=eval(parse(text=paste("mean_",imputation_metric,sep="")))-eval(parse(text=paste("sd_",imputation_metric,sep=""))),
      #                     ymax=eval(parse(text=paste("mean_",imputation_metric,sep="")))+eval(parse(text=paste("sd_",imputation_metric,sep="")))),
      #                 width=.2, position=position_dodge(.9),color="black") +
      #   labs(title=bquote( list("n =" ~ .(N) ~ ", p =" ~ .(P) ~ ", d =" ~ .(D), ~ mu[phi]==.(phi0)) ), y = imputation_metric0, x="Method") +
      #   theme(legend.title = element_blank(), text=element_text(size = 20),axis.text.x = element_text(colour = c(rep("black",nlevels(bar_df$method0)-1),"red")))

      ##### Group by mechanism, color by method
      p=ggplot(bar_df,aes(x=mechanism, y=eval(parse(text=paste("mean_",imputation_metric,sep=""))), fill=method, color=method))+
        geom_bar(stat="identity",position=position_dodge(.9),alpha=0.4,color="black")+#ylim(c(0,3))+#ylim(c(0,0.5))+
        geom_errorbar(aes(ymin=eval(parse(text=paste("mean_",imputation_metric,sep="")))-eval(parse(text=paste("sd_",imputation_metric,sep=""))),
                          ymax=eval(parse(text=paste("mean_",imputation_metric,sep="")))+eval(parse(text=paste("sd_",imputation_metric,sep="")))),
                      width=.2, position=position_dodge(.9),color="black") +
        labs(title=bquote( list("n =" ~ .(N) ~ ", p =" ~ .(P) ~ ", d =" ~ .(D), ~ mu[phi]==.(phi0)) ), y = imputation_metric0, x="Mechanism") +
        theme(legend.title = element_blank(), text=element_text(size = 20))#,axis.text.x = element_text(colour = c(rep("black",nlevels(bar_df$mechanism)-1),"red")))


      ggsave(filename=sprintf("%s/%s_5sims_miss%d.png",dir_name2,imputation_metric,miss_pct),
             plot=p, width = 12, height=7, units="in")
    }
    }}}
}

save_MI=function(dataset, mechanism, miss_pct, phi0, sim_index, P, methods=c("MICE_sup","MICE_unsup","NIMIWAE_sup","NIMIWAE_unsup"), init="default"){

  # P0 = if(!intercept){P}else{P+1}
  iP = P+1
  runs=length(sim_index)

  ires <- array(NA, dim = c(length(methods), runs, iP, 4))
  dimnames(ires) <- list(methods,
                         as.character(1:runs),
                         paste("x",1:iP,sep=""),
                         c("estimate", "std.error", "2.5 %","97.5 %"))
  res <- array(NA, dim = c(length(methods), runs, P, 4))
  dimnames(res) <- list(methods,
                        as.character(1:runs),
                        paste("x",1:P,sep=""),
                        c("estimate", "std.error", "2.5 %","97.5 %"))
  m=50
  # all_AUCs = array(NA, dim = c(length(methods),runs,m))
  ## Run each mice method and extract coef estimates
  # trues = list(); fitted=list()
  trues = matrix(nrow=length(sim_index), ncol=P); fitteds = matrix(nrow=length(sim_index), ncol=P)
  itrues = matrix(nrow=length(sim_index), ncol=iP); ifitteds = matrix(nrow=length(sim_index), ncol=iP)
  a=1;b=1;c=1;d=1;e=1

  # if(!intercept){name1 = "MI"} else{ name1="MI0" }
  # if(compare=="truth"){name1 = sprintf("%s",name1)} else if(compare=="est"){name1 = sprintf("e%s",name1)}

  if(init=="default"){NIM_pref = ""} else if(init=="alt"){NIM_pref="/alt_init"}
  res_file = sprintf("./Results/%s/phi%d%s/MI_res_%s_%d.RData",
                     dataset, NIM_pref, phi0, mechanism, miss_pct)
  res_file2 = sprintf("./Results/%s/phi%d%s/MI_results_%s_%d.RData",
                      dataset, NIM_pref, phi0, mechanism, miss_pct)

  # if(!file.exists(res_file)){
  for(c in 1:length(sim_index)){
    fits = list()
    data.file.name=sprintf("Results/%s/phi%d/sim%d/data_%s_%d.RData",
                           dataset, phi0, sim_index[c], mechanism, miss_pct)

    # rownames(all_AUCs) = methods
    for(i in 1:length(methods)){
      dir_name0 = if(grepl("^IMIWAE",methods[i])){
        sprintf("Results/%s/phi%d/sim%d/Ignorable",dataset, phi0, sim_index[c], mechanism, miss_pct)  # Ignorable NIMIWAE
      } else if(grepl("^NIMIWAE",methods[i])){ sprintf("Results/%s/phi%d/sim%d%s",dataset, phi0, sim_index[c],NIM_pref)
      } else { sprintf("Results/%s/phi%d/sim%d",dataset, phi0, sim_index[c]) }

      if(grepl("^IMIWAE",methods[i])){ method0 = sprintf("N%s",methods[i]) } else{ method0 = methods[i] }  # IMIWAE is labeled NIMIWAE too, except in Ignorable directory

      if(grepl("IMIWAE",methods[i])){
        file.name=sprintf("%s/res_%s_%s_%d_IWAE_rzF.RData",dir_name0, method0, mechanism, miss_pct)
      }else{
        file.name=sprintf("%s/res_%s_%s_%d.RData",dir_name0, method0, mechanism, miss_pct)
      }
      dir_name = if(grepl("_sup",method0)){
        sprintf("%s/%s_sup/miss%d",dir_name0, mechanism, miss_pct)
      } else{
        sprintf("%s/%s_unsup/miss%d",dir_name0, mechanism, miss_pct)
      }
      if(!file.exists(file.name)){
        next
      } else{print(file.name)}
      fit = process_results_MI(data.file.name, file.name, method0, dir_name)
      res[i, c, , ] <- as.matrix(fit$tab)
      ires[i, c, , ] <- as.matrix(fit$itab)
      # rownames(res[i,c,,]) = fit$names
      print(res[i,c,,])
      print(ires[i, c, , ])
      names=fit$names
      true_beta = fit$fit_data$params$beta
      itrue_beta = c(fit$fit_data$params$beta0, fit$fit_data$params$beta)

      fitted_beta = fit$trues
      ifitted_beta = fit$itrues
      # true = fit$betas
      rm(fit); gc()
    }
    trues[c,] = if(!is.null(true_beta)){ true_beta }else{NA}
    itrues[c,] = if(!is.null(itrue_beta)){ itrue_beta }else{NA}
    fitteds[c,] = fitted_beta
    ifitteds[c,] = ifitted_beta
    print(c); print(trues); print(itrues); print(fitteds); print(ifitteds)
  }


  ests = t(apply(array(res[,,,1], dim=c(length(methods), runs, P)),c(1,3),function(x)mean(x,na.rm=T))); SEs = t(apply(array(res[,,,2],dim=c(length(methods), runs, iP)),c(1,3),function(x)mean(x,na.rm=T)))
  iests = t(apply(array(ires[,,,1], dim=c(length(methods), runs, P)),c(1,3),function(x)mean(x,na.rm=T))); iSEs = t(apply(array(ires[,,,2], dim=c(length(methods), runs, iP)),c(1,3),function(x)mean(x,na.rm=T)))

  # rownames(ests) = names; rownames(SEs) = names; rownames(iests) = c("Intercept",names); rownames(iSEs) = c("Intercept",names)
  colnames(ests) = methods; colnames(SEs) = methods; colnames(iests) = methods; colnames(iSEs) = methods

  save(list=c("res","ires","names","ests", "iests","SEs","iSEs","trues","fitteds", "itrues","ifitteds"), file=res_file)
  print("saved res")
  ## Compute performance metrics
  results = list(); all_results = list()  # all_results when there are more than 1 rep --> save all rep and features
  iresults = list(); iall_results = list()  # all_results when there are more than 1 rep --> save all rep and features
  # true = trues[[1]]   # trues: runs x P (or P+1); res: methods x runs x P x metrics
  for(j in 1:iP){
    print("j:"); print(j)
    if(j<iP){
      true = matrix(trues[,j],byrow=T,nrow=length(methods),ncol=runs)
      fitted = matrix(fitteds[,j],byrow=T,nrow=length(methods),ncol=runs)
    }

    itrue = matrix(itrues[,j],byrow=T,nrow=length(methods),ncol=runs)
    ifitted = matrix(ifitteds[,j],byrow=T,nrow=length(methods),ncol=runs)
    out_metrics = function(res,true,fitted,j){
      RBs <- res[,,j, "estimate"] - true
      ABs <- abs(RBs)
      PBs <- 100 * abs((res[,,j, "estimate"] - true)/ true)
      CRs <- (res[,,j, "2.5 %"] < true & true < res[,,j, "97.5 %"])^2
      AWs <- res[,,j, "97.5 %"] - res[,,j, "2.5 %"]
      RMSEs <- sqrt((res[,,j, "estimate"] - true)^2)
      NRMSEs <- sqrt((res[,,j, "estimate"] - true)^2/abs(true))

      eRBs <- res[,,j, "estimate"] - fitted
      eABs <- abs(eRBs)
      ePBs <- 100 * abs((res[,,j, "estimate"] - fitted)/ fitted)
      eCRs <- (res[,,j, "2.5 %"] < fitted & fitted < res[,,j, "97.5 %"])^2
      eAWs <- res[,,j, "97.5 %"] - res[,,j, "2.5 %"]
      eRMSEs <- sqrt((res[,,j, "estimate"] - fitted)^2)
      eNRMSEs <- sqrt((res[,,j, "estimate"] - fitted)^2/abs(fitted))

      return(list(RB=RBs,AB=ABs,PB=PBs,CR=CRs,AW=AWs,RMSE=RMSEs,NRMSE=NRMSEs,
                  eRB=eRBs,eAB=eABs,ePB=ePBs,eCR=eCRs,eAW=eAWs,eRMSE=eRMSEs,eNRMSE=eNRMSEs))
    }

    if(length(sim_index)==1){

      if(j<iP){
        print(true)
        print(res[,,j, "estimate"])
        fit = out_metrics(res,true,fitted,j)
        RB=fit$RB; AB=fit$AB; PB=fit$PB; CR=fit$CR; AW=fit$AW; RMSE=fit$RMSE; NRMSE=fit$NRMSE
        eRB=fit$eRB; eAB=fit$eAB; ePB=fit$ePB; eCR=fit$eCR; eAW=fit$eAW; eRMSE=fit$eRMSE; fit$eNRMSE
        all_results[[j]] = data.frame(RB=c(RB),AB=c(AB),PB=c(PB),CR=c(CR),AW=c(AW),RMSE=c(RMSE), #NRMSE=c(NRMSE),
                                      eRB=c(eRB),eAB=c(eAB),ePB=c(ePB),eCR=c(eCR),eAW=c(eAW),eRMSE=c(eRMSE), #eNRMSE=c(eNRMSE),
                                      method=methods, rep=sim_index)
      }

      print(ires[,,j, "estimate"])

      fit = out_metrics(ires,itrue,ifitted,j)
      iRB=fit$RB; iAB=fit$AB; iPB=fit$PB; iCR=fit$CR; iAW=fit$AW; iRMSE=fit$RMSE; iNRMSE=fit$NRMSE
      ieRB=fit$eRB; ieAB=fit$eAB; iePB=fit$ePB; ieCR=fit$eCR; ieAW=fit$eAW; ieRMSE=fit$eRMSE; ieNRMSE=fit$eNRMSE
      iall_results[[j]] = data.frame(RB=c(iRB),AB=c(iAB),PB=c(iPB),CR=c(iCR),AW=c(iAW),RMSE=c(iRMSE), NRMSE=c(iNRMSE),
                                     eRB=c(ieRB),eAB=c(ieAB),ePB=c(iePB),eCR=c(ieCR),eAW=c(ieAW),eRMSE=c(ieRMSE), eNRMSE=c(ieNRMSE),
                                     method=methods, rep=sim_index)
    }else{
      if(j<iP){
        print(true)
        print(res[,,j, "estimate"])
        fit = out_metrics(res,true,fitted,j)
        RBs=fit$RB; ABs=fit$AB; PBs=fit$PB; CRs=fit$CR; AWs=fit$AW; RMSEs=fit$RMSE; NRMSEs=fit$NRMSE
        eRBs=fit$eRB; eABs=fit$eAB; ePBs=fit$ePB; eCRs=fit$eCR; eAWs=fit$eAW; eRMSEs=fit$eRMSE; eNRMSEs=fit$eNRMSE

        print(RBs)

        RB <- rowMeans(RBs,na.rm = T)
        AB <- rowMeans(ABs,na.rm = T)
        PB <- rowMeans(PBs,na.rm = T)
        CR <- rowMeans(CRs,na.rm = T)
        AW <- rowMeans(AWs,na.rm = T)
        RMSE <- sqrt(rowMeans((res[,,j, "estimate"] - true)^2,na.rm = T))
        NRMSE <- sqrt(rowMeans((res[,,j, "estimate"] - true)^2/abs(true),na.rm = T))   # normalized by the value of the truth
        eRB <- rowMeans(eRBs,na.rm = T)
        eAB <- rowMeans(eABs,na.rm = T)
        ePB <- rowMeans(ePBs,na.rm = T)
        eCR <- rowMeans(eCRs,na.rm = T)
        eAW <- rowMeans(eAWs,na.rm = T)
        eRMSE <- sqrt(rowMeans((res[,,j, "estimate"] - fitted)^2,na.rm = T))
        eNRMSE <- sqrt(rowMeans((res[,,j, "estimate"] - fitted)^2/abs(fitted),na.rm = T))   # normalized by the value of the truth

        all_results[[j]] = data.frame(RB=c(RBs),AB=c(ABs),PB=c(PBs),CR=c(CRs),AW=c(AWs),RMSE=c(RMSEs), NRMSE=c(NRMSEs),
                                      eRB=c(eRBs),eAB=c(eABs),ePB=c(ePBs),eCR=c(eCRs),eAW=c(eAWs),eRMSE=c(eRMSEs), eNRMSE=c(eNRMSEs),
                                      method=rep(methods, times=runs), rep=rep(sim_index, each=length(methods)))
      }

      print(ires[,,j, "estimate"])
      fit = out_metrics(ires,itrue,ifitted,j)
      iRBs=fit$RB; iABs=fit$AB; iPBs=fit$PB; iCRs=fit$CR; iAWs=fit$AW; iRMSEs=fit$RMSE; iNRMSEs=fit$NRMSE
      ieRBs=fit$eRB; ieABs=fit$eAB; iePBs=fit$ePB; ieCRs=fit$eCR; ieAWs=fit$eAW; ieRMSEs=fit$eRMSE; ieNRMSEs=fit$eNRMSE

      iRB <- rowMeans(iRBs,na.rm = T)
      iAB <- rowMeans(iABs,na.rm = T)
      iPB <- rowMeans(iPBs,na.rm = T)
      iCR <- rowMeans(iCRs,na.rm = T)
      iAW <- rowMeans(iAWs,na.rm = T)
      iRMSE <- sqrt(rowMeans((ires[,,j, "estimate"] - itrue)^2,na.rm = T))
      iNRMSE <- sqrt(rowMeans((ires[,,j, "estimate"] - itrue)^2/abs(itrue),na.rm = T))   # normalized by the value of the truth
      ieRB <- rowMeans(ieRBs,na.rm = T)
      ieAB <- rowMeans(ieABs,na.rm = T)
      iePB <- rowMeans(iePBs,na.rm = T)
      ieCR <- rowMeans(ieCRs,na.rm = T)
      ieAW <- rowMeans(ieAWs,na.rm = T)
      ieRMSE <- sqrt(rowMeans((ires[,,j, "estimate"] - ifitted)^2,na.rm = T))
      ieNRMSE <- sqrt(rowMeans((ires[,,j, "estimate"] - ifitted)^2/abs(ifitted),na.rm = T))   # normalized by the value of the truth
      iall_results[[j]] = data.frame(RB=c(iRBs),AB=c(iABs),PB=c(iPBs),CR=c(iCRs),AW=c(iAWs),RMSE=c(iRMSEs), NRMSE=c(iNRMSEs),
                                     eRB=c(ieRBs),eAB=c(ieABs),ePB=c(iePBs),eCR=c(ieCRs),eAW=c(ieAWs),eRMSE=c(ieRMSEs), eNRMSE=c(ieNRMSEs),
                                     method=rep(methods, times=runs), rep=rep(sim_index, each=length(methods)))
    }
    # results[[j]] = data.frame(RB, PB, CR, AW, RMSE)
    if(j<iP){
      results[[j]] = data.frame(RB=RB, AB=AB, PB=PB, CR=CR, AW=AW, RMSE=RMSE, NRMSE=NRMSE,
                                eRB=eRB, eAB=eAB, ePB=ePB, eCR=eCR, eAW=eAW, eRMSE=eRMSE, eNRMSE=eNRMSE)
    }
    iresults[[j]] = data.frame(RB=iRB, AB=iAB, PB=iPB, CR=iCR, AW=iAW, RMSE=iRMSE, NRMSE=iNRMSE,
                               eRB=ieRB, eAB=ieAB, ePB=iePB, eCR=ieCR, eAW=ieAW, eRMSE=ieRMSE, eNRMSE=ieNRMSE)
  }
  # return(list(res=res, true=true, results=results, times=times))
  print("Averaged results:")
  print( Reduce("+",results)/length(results) )
  print( Reduce("+",iresults)/length(iresults) )

  df = data.frame(do.call(rbind, results))
  idf = data.frame(do.call(rbind, iresults))
  all_df = data.frame(do.call(rbind, all_results))
  iall_df = data.frame(do.call(rbind, iall_results))

  df$method=methods; idf$method=methods
  df$feature=rep(c(1:P),each=length(methods)); idf$feature=rep(c(1:iP),each=length(methods))
  df$feature = as.factor(df$feature); idf$feature = as.factor(idf$feature)

  all_df$feature=rep(c(1:P),each=length(methods)*runs); iall_df$feature=rep(c(1:iP),each=length(methods)*runs)
  all_df$feature = as.factor(all_df$feature); iall_df$feature = as.factor(iall_df$feature)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(gtable)

  save(list=c("results","all_results", "iresults","iall_results"), file=res_file2)

  ### MI table of results, Excluding intercept
  p = tableGrob(round(Reduce("+",results)/length(results), 3))
  title1 <- textGrob("Results MI (Mean Across Features), No Int",gp=gpar(fontsize=8))
  padding <- unit(5,"mm")
  p <- gtable_add_rows(
    p,
    heights = grobHeight(title1) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p,
    title1,
    1, 1, 1, ncol(p))
  ggfname = sprintf("./Results/%s/phi%d%s/MI_results_%s_%d.png",
                    dataset,phi0,NIM_pref,mechanism,miss_pct)
  ggsave(p,file=ggfname, width=16, height=10, units="in")

  ### MI table of results, Including intercept
  p = tableGrob(round(Reduce("+",iresults)/length(iresults), 3))
  title1 <- textGrob("Results MI (Mean Across Features), Int",gp=gpar(fontsize=8))
  padding <- unit(5,"mm")
  p <- gtable_add_rows(
    p,
    heights = grobHeight(title1) + padding,
    pos = 0)
  p <- gtable_add_grob(
    p,
    title1,
    1, 1, 1, ncol(p))
  ggfname = sprintf("./Results/%s/phi%d%s/iMI_results_%s_%d.png",
                    dataset, phi0,NIM_pref,mechanism,miss_pct)
  ggsave(p,file=ggfname, width=16, height=10, units="in")


  metrics = c("RB","AB","PB","CR","AW","RMSE","NRMSE",
              "eRB","eAB","ePB","eCR","eAW","eRMSE","eNRMSE")
  for(i in 1:length(metrics)){
    print(paste("Plotting",metrics[i]))
    p = ggplot(data=df, aes(x=feature, y=eval(parse(text=metrics[i])), fill=method)) +
      geom_bar(stat="identity",position=position_dodge(), width=.5) +
      labs(title=sprintf("P=%d, %s, %s vs Feature", P,mechanism, metrics[i]), y=metrics[i])
    ggfname = sprintf("./Results/%s/phi%d%s/MI_%s_%s_%d.png",
                      dataset, phi0,NIM_pref,metrics[i], mechanism, miss_pct)
    ggsave(filename=ggfname,
           plot=p, height=6, width=9)

    p = ggplot(data=idf, aes(x=feature, y=eval(parse(text=metrics[i])), fill=method)) +
      geom_bar(stat="identity",position=position_dodge(), width=.5) +
      labs(title=sprintf("P=%d, %s, %s vs Feature", iP,mechanism, metrics[i]), y=metrics[i])
    ggfname = sprintf("./Results/%s/phi%d%s/iMI_%s_%s_%d.png",
                      dataset, phi0,NIM_pref,metrics[i], mechanism, miss_pct)
    ggsave(filename=ggfname,
           plot=p, height=6, width=9)
  }

}


tabulate_UCI=function(datasets=c("RED","CONCRETE","BREAST","BANKNOTE","WHITE","YEAST","IRIS",
                                 "HEPMASS","GAS","POWER","MINIBOONE"),
                      mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(25),
                      methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi0=5,sim_index=1,
                      arch=NULL, rdeponz=NULL){
  # return df with columns: dataset, method, mechanism, and value (NRMSE, L1, L2, MSE, or RMSE. choose here)
  nvals = length(datasets)*length(mechanisms)*length(miss_pct)*length(methods)*length(sim_index)
  df=data.frame(matrix(nrow=nvals,ncol=10))
  index=1
  colnames(df) = c("dataset","mechanism","method","miss_pct","sim_index","MSE","RMSE","NRMSE","L1","L2")
  for(i in 1:length(datasets)){for(j in 1:length(mechanisms)){for(k in 1:length(methods)){for(l in 1:length(miss_pct)){for(m in 1:length(sim_index)){
    ### LOAD Results/datasets[i]/phi5/sim(index)/res_(method)_(mechanism)_(miss_pct).RData
    # if(methods[k]=="NIMIWAE2"){subdir="/Ignorable"; method0="NIMIWAE"}else{subdir=""; method0=methods[k]}

    subdir=""; method0=methods[k]
    dir_name0 = sprintf("Results/%s/phi%d/sim%d",datasets[i],phi0,sim_index[m])
    dir_name = sprintf("%s%s",dir_name0,subdir)
    data.file.name=sprintf("%s/data_%s_%d.RData",dir_name0,mechanisms[j],miss_pct[l])
    file.name = output_file.name(dir_name=dir_name,method=method0, mechanism=mechanisms[j],
                                 miss_pct=miss_pct[l], arch=arch, rdeponz=rdeponz)
    df[index,1:5] = c(datasets[i],mechanisms[j],methods[k],miss_pct[l],sim_index[m])
    print(data.file.name)
    print(file.name)
    if(file.exists(file.name)){
      res = process_results(data.file.name,file.name,method0)$results   # returns unlist(list(MSE=MSE,RMSE=RMSE,NRMSE=NRMSE,L1=L1,L2=L2))
      df[index,6:10] = res
    } else{ df[index,6:10] = NA }
    index = index + 1
  }}}}}
  df$dataset = as.factor(df$dataset); df$mechanism = as.factor(df$mechanism); df$method = as.factor(df$method)
  df$miss_pct = as.factor(df$miss_pct); df$sim_index = as.factor(df$sim_index)

  df$MSE = as.numeric(df$MSE); df$NRMSE = as.numeric(df$NRMSE); df$RMSE = as.numeric(df$RMSE);
  df$L1 = as.numeric(df$L1); df$L2 = as.numeric(df$L2)
  return(df)
}


# DONE ######### NEED TO ADD MI FUNCTIONS FOR SIM AND FOR PHYSIONET
######### NEED TO ADD FIGURES FOR MI FOR SIM
######### NEED TO SAVE MI RESULTS FOR PHYSIONET TO Results/Physionet/phiNA/MI_res_MNAR_NA.RData
######### NEED TO REDO THIS FUNCTION FOR PHYSIONET
summarize_Physionet = function(prefix="",mechanism="MNAR",miss_pct=NA,sim_index=1,dataset="Physionet", init="alt"){
  phi0 = NA; D=NA
  NIM_pref = if(init=="default"){""}else if(init=="alt"){"alt_init/"}

  # df = data.frame()
  dir_name = sprintf("Results/%s/phi%d/Z%d",dataset,phi0,D)
  load(sprintf("%s/sim%d/data_MNAR_%d.RData", dir_name,sim_index, miss_pct))
  load(sprintf("%s/%sMI_res_%s_%d.RData",dir_name,NIM_pref,mechanism,miss_pct))    # 100% train --> just inference (no predictions)
  rownames(ests) = colnames(data)
  df = ests[order(-abs(ests[,1])),]; df2 = SEs[order(-abs(ests[,1])),]
  pheatmap::pheatmap(df,cluster_rows=F,scale="row",cluster_cols=F)

  library(grid); library(gridExtra)
  n=20
  tab = matrix(paste( round(df[1:n,],3), " (",round(df2[1:n,],3), ")", sep=""), nrow=n)
  rownames(tab) = rownames(df)[1:n]; colnames(tab)=colnames(df)
  grid.table(tab)



  ratios=c(train = .8, valid = .2)
  set.seed(333)
  g = sample(cut(
    seq(nrow(data)),
    nrow(data)*cumsum(c(0,ratios)),
    labels = names(ratios)
  ))

  classes = fit_data$classes

  P=ncol(data); N=nrow(data)

  # if(grepl("_sup",method)){
  #   ## supervised methods
  #   data = cbind(data, classes)
  #   Missing = cbind(Missing, rep(1, nrow(Missing)))  # include 1's for classes (fully observed)
  # }

  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)
  classess = split(classes,g)

  datas$test = datas$train
  Missings$test = Missings$train
  classess$test = classess$train

  # id = 95  # FiO2_last
  # id = 27  # RespRate_last
  for(id in 1:ncol(data)){
    if(all(Missings$test[,id]==1)){next}
    tab = table(Missings$test[,id], classess$test)
    tab = tab/sum(tab)

    obs_p = c(mean(Missings$test[classess$test==1,id]), mean(Missings$test[classess$test==0,id]))  ## less observed for class = 0 (no mortality)

    df = data.frame(obs_p = obs_p, mortality=c("Deceased","Survived"))

    p = ggplot(df,aes(y=obs_p,x=mortality)) + geom_bar(stat="identity",position=position_dodge(), width=.5) +
      ggtitle(sprintf("Proportion of Non-missing Observations of %s",colnames(data)[id]), subtitle = "Stratified by In-Hospital Mortality") +
      labs(x="In-Hospital Mortality", y="Proportion of Non-missing Observations")
    ggsave(filename=sprintf("%s/sim1/%sDiagnostics/%s_col%d_missingness.png", dir_name,NIM_pref,colnames(data)[id],id), plot=p,
           width=6, height=6, units="in")
  }


  #################################################

  order_idx = order(-abs(ests[,"NIMIWAE_sup"]))
  head(ests[order_idx,], n=10)

  dir_name2 = sprintf("Results/%s/phi%d/Z%d/sim%d",dataset,phi0,D,sim_index)



  library(ggplot2)
  library(ggfortify)
  library(gridExtra)

  load(sprintf("%s/%sres_NIMIWAE_sup_MNAR_NA_IWAE_rzF.RData",dir_name2,NIM_pref))
  res0 = res_NIMIWAE
  xhat0 = res_NIMIWAE$xhat_rev
  load(sprintf("%s/%sres_NIMIWAE_unsup_MNAR_NA_IWAE_rzF.RData",dir_name2,NIM_pref))
  res1 = res_NIMIWAE
  xhat1 = res_NIMIWAE$xhat_rev
  load(sprintf("%s/Ignorable/res_NIMIWAE_sup_MNAR_NA_IWAE_rzF.RData",dir_name2))
  res2 = res_NIMIWAE
  xhat2 = res_NIMIWAE$xhat_rev
  load(sprintf("%s/Ignorable/res_NIMIWAE_unsup_MNAR_NA_IWAE_rzF.RData",dir_name2))
  res3 = res_NIMIWAE
  xhat3 = res_NIMIWAE$xhat_rev
  # }else if(method=="MIWAE"){         # AUROC = 0.7808
  load(sprintf("%s/res_MIWAE_MNAR_NA.RData",dir_name2))
  xhat4 = res_MIWAE$xhat_rev
  # }else if(method=="HIVAE"){
  load(sprintf("%s/res_HIVAE_MNAR_NA.RData",dir_name2))
  xhat5 = res_HIVAE$data_reconstructed
  # }else if(method=="VAEAC"){         # AUROC = 0.7686
  load(sprintf("%s/res_VAEAC_MNAR_NA.RData",dir_name2))
  xhat_all = res_VAEAC$result
  # average imputations
  xhat6 = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
  n_imputations = res_VAEAC$train_params$n_imputations
  for(i in 1:nrow(datas$test)){
    xhat6[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
  }
  # }else if(method=="MEAN"){
  # load(sprintf("%s/res_MEAN_%s_%d.RData",dir_name2,mechanism,miss_pct))
  # xhat5 = res_MEAN$xhat_rev         # AUROC = 0.7787

  xhat7 = matrix(apply(datas$train,2,function(x)mean(x,na.rm=T)),byrow=T,nrow=nrow(datas$test), ncol=ncol(datas$test))
  # }else if(method=="MF"){
  load(sprintf("%s/res_MF_MNAR_NA.RData",dir_name2))
  xhat8 = res_MF$xhat_rev         # AUROC = 0.7635
  # }
  load(sprintf("%s/res_MICE_unsup_MNAR_NA.RData",dir_name2))
  xhat9 = res_MICE$xhat         # AUROC = 0.7635
  # }
  load(sprintf("%s/res_MICE_sup_MNAR_NA.RData",dir_name2))
  xhat10 = res_MICE$xhat         # AUROC = 0.7635
  # }

  # 0: NIMIWAE_Ignorable, 1: NIMIWAE, 2: MIWAE, 3: HIVAE
  # 4: VAEAC, 5: MEAN, 6: MF

  for(ii in 1:ncol(data)){
    idy=ii
    if(all(Missings$test[,idy]==1)){next}


    idx = Missings$test[,idy]==0
    df = data.frame(c(xhat0[idx,idy],xhat1[idx,idy],xhat2[idx,idy],xhat3[idx,idy],
                      xhat4[idx,idy],xhat5[idx,idy],xhat6[idx,idy],xhat8[idx,idy],xhat9[idx,idy],xhat10[idx,idy]))
    df$method = rep(c("NIMs","NIMu","IMs","IMu","MIWAE","HIVAE","VAEAC","MF","MICEu","MICEs"), each = sum(idx))
    names(df) = c("value","method")
    df$value = as.numeric(df$value)
    p = ggplot(df, aes(x=method,y=value)) + geom_boxplot(fill="gray",outlier.shape=NA) + scale_y_continuous(limits = quantile(df$value, c(0.05, 0.95))) +
      ggtitle(sprintf("Boxplot of Imputed Values of %s",colnames(data)[idy])) + labs(x="Method",y="Imputed Value") + geom_hline(yintercept=xhat7[Missings$test[,idy]==0,idy][1], color="red")
    ggsave(filename=sprintf("%s/%sDiagnostics/%s_col%d.png", dir_name2,NIM_pref,colnames(data)[idy],idy), plot=p,
           width=6, height=6, units="in")
    # print(p)

  }

}
