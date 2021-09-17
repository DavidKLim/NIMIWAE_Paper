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

output_file.name=function(dir_name,method=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),
                          mechanism,miss_pct,arch,rdeponz){
  if(method=="NIMIWAE"){
    yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    file.name=sprintf("%s/res_NIMIWAE_%s_%d_%s_%s",
                      dir_name,method,mechanism,miss_pct,arch,yesrdeponz)
  }else if(method=="IMIWAE"){
    yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    file.name=sprintf("%s/res_NIMIWAE_%s_%d_%s_%s_ignorable",
                      dir_name,method,mechanism,miss_pct,arch,yesrdeponz)
  }else{
    file.name=sprintf("%s/res_%s_%s_%d",
                      dir_name,method,mechanism,miss_pct) # for miwae, default = Normal (StudentT can be done later)
  }
  print(file.name)
  file.name=sprintf("%s.RData",file.name)
  return(file.name)
}

process_results=function(data.file.name, file.name, method=c("MIWAE","IMIWAE","NIMIWAE","HIVAE","VAEAC","MEAN","MF","MICE")){
  call_name=match.call()

  # load data and split into training/valid/test sets
  load(data.file.name)
  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  Missings = split(data.frame(Missing), g)

  if(grepl("TOY", data.file.name) | grepl("SIM",data.file.name)){
    classes = fit_data$classes
  }
  classess = split(classes,g)
  if(is.null(datas$test)){
    datas$test = datas$train; Missings$test = Missings$train; classess$test = classess$train     # if no test split (no custom split) --> train set is test set
  }

  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  # MIWAE and NIMIWAE only
  load(file.name)
  print(file.name)
  fit=eval(parse(text=paste("res",method,sep="_")))
  if(method=="IMIWAE"){fit=res_NIMIWAE}

  #xhat=reverse_norm_MIWAE(fit$xhat,norm_means,norm_sds)   # already reversed
  if(method %in% c("MIWAE","NIMIWAE","IMIWAE")){
    # xhat=fit$xhat_rev
    xhat=fit$xhat
  }else if(method =="HIVAE"){
    xhat=fit$data_reconstructed
  }else if(method=="VAEAC"){
    xhat_all = fit$result    # this method reverses normalization intrinsically
    # average imputations
    xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
    n_imputations = fit$train_params$n_imputations
    for(i in 1:nrow(datas$test)){
      xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
    }
  }else if(method=="MEAN"){
    # xhat = fit$xhat_rev
    if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mean,norm_means,norm_sds)}
    xhat = fit$xhat_rev
  }else if(method=="MF"){
    # xhat = fit$xhat_rev
    if(is.null(fit$xhat_rev)){fit$xhat_rev = reverse_norm_MIWAE(fit$xhat_mf,norm_means,norm_sds)}
    xhat = fit$xhat_rev
  }else if(method=="MICE"){
    # library(mice)
    # miss_ids = apply(Missings$test, 2, mean)!=1   # not fully observed ids
    # fit$imp[miss_ids]
    #
    # xhat = fit$xhat
    # imp_values = sapply(fit$imp, rowMeans)  ### imputed values averaged across MI's: all cols
    #
    # for(c in 1:length(miss_ids)){
    #   if(miss_ids[c]){ xhat[Missings$test[,c]==0,c] = imp_values[[c]]
    #   } else{next}
    # }

    # list_xhats = list()
    # for(ii in 1:res_MICE$m){   # default at m=5 imputations by mice
    #   list_xhats[[ii]] = complete(fit,ii)
    # }
    #
    # xhat = Reduce("+", list_xhats)/length(list_xhats)

    xhat=fit$xhat
  }

  # check same xhat:
  print("Mean Squared Error (Observed): should be 0")
  print(mean((xhat[Missings$test==1] - datas$test[Missings$test==1])^2))    # should be 0
  print("Mean Squared Error (Missing):")
  print(mean((xhat[Missings$test==0] - datas$test[Missings$test==0])^2))

  # Imputation metrics

  imputation_metrics=NRMSE(x=datas$test, xhat=xhat, Missing=Missings$test)
  #imputation_metrics=NRMSE(x=xfull, xhat=xhat, Missing=Missings$test)

  predict_classes = function(X_train, y_train, X_test, y_test, family="binomial"){
    if(family=="binomial"){
      # logistic regression --> y_train and y_test have to be factors
      y_train = as.factor(y_train)
      y_test = as.factor(y_test)
    }
    data_train = data.frame(cbind(X_train, y_train))
    fit = glm(y_train ~ 1 + X_train, family=family)
    y_predicted = predict.glm(fit, newdata=data.frame(X_test))

    return(list(fit_train = fit, y_predicted = y_predicted,
                acc = mean(y_predicted==y_test)))
  }
  ratio = 0.8; n_train = floor(ratio*nrow(xhat)); n_test = nrow(xhat) - n_train
  idx = c( rep(T, n_train) , rep(F, n_test) )

  fit_pred_imputed = predict_classes(X_train=as.matrix(xhat[idx,]), y_train=classess$test[idx], X_test=as.matrix(xhat[!idx,]), y_test=classess$test[!idx])
  fit_pred_true = predict_classes(X_train=as.matrix(datas$test[idx,]), y_train=classess$test[idx], X_test=as.matrix(datas$test[!idx,]), y_test=classess$test[!idx])

  fits_pred = list(imputed = fit_pred_imputed,
                   true = fit_pred_true)
  fit$fits_pred = fits_pred


  results = c(unlist(imputation_metrics))
  return(list(fit=fit,results=results,call=call_name))
}

processComparisons=function(dir_name="Results/SIM1/phi5",mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(15,25,35),
                            methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),
                            imputation_metric=c("MSE","NRMSE","L1","L2"), arch=NULL, rdeponz=NULL, outfile=NULL){
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(reshape2)
  g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)}
  mats_res = list(); mats_params=list()
  print("Compiling results...")
  list_res = list()
  for(ii in 1:length(miss_pct)){
    params=list()
    # input true probs, no learning R, Normal distrib, input_r="r", vary rdeponz and sample_r (4)
    index=1

    for(i in 1:length(mechanisms)){for(j in 1:length(methods)){
      data.file.name=sprintf("%s/data_%s_%d.RData",dir_name,mechanisms[i],miss_pct[ii])
      file.name = output_file.name(dir_name=dir_name,method=methods[j], mechanism=mechanisms[i],
                                   miss_pct=miss_pct[ii], arch=arch, rdeponz=rdeponz)
      list_res[[index]]=process_results(data.file.name,file.name,methods[j])
      params[[index]]=c(methods[j],mechanisms[i],miss_pct[ii])
      names(params[[index]])=c("method","mechanism","miss_pct")
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
    other_methods = methods[methods!="NIMIWAE"]; other_methods[other_methods=="MEAN"]="Mean"; other_methods[other_methods=="MF"]="MissForest"
    df_bar$method = factor(df_bar$method,
                           levels=c(other_methods[order(other_methods)],"NIMIWAE"))
    # levels(df_bar$method)=methods[order(methods)]

    gg_color_hue <- function(n) {
      hues = seq(15, 375, length = n + 1)
      hcl(h = hues, l = 65, c = 100)[1:n]
    }
    colors = gg_color_hue(length(methods))

    p=ggplot(df_bar,aes(x=method,y=eval(parse(text=imputation_metric)),fill=mechanism,color=mechanism))+
      geom_bar(stat="identity",position=position_dodge(.9),alpha=0.4)+#ylim(c(0,3))+#ylim(c(0,0.5))+
      labs(title=sprintf("%s vs cases",imputation_metric),
           subtitle = "Imputation performance across missingness mechanisms",
           y = imputation_metric, x="Method")+
      theme(text=element_text(size = 20)) #+
    # scale_color_manual(breaks=methods[order(methods)],
    #                    values=colors[1:length(methods)])+scale_fill_manual(values=colors[1:length(methods)])

    if(is.null(outfile)){
      png(sprintf("%s/%s_competing_miss%d.png",dir_name,imputation_metric,miss_pct[ii]),width=1200,height=500)
    } else{ png(outfile, width=1200, height=500)}
    #barplot(mat_res[rownames(mat_res)=="NRMSE",])
    print(p)
    dev.off()

    # save in mats_res
    mats_res[[ii]]=mat_res
    mats_params[[ii]]=mat_params
  }
  names(mats_res)=miss_pct
  return(list(res=mats_res, params=mats_params))
}

saveFigures = function(datasets=c("SIM1","SIM2","SIM3"), sim_index=1:5, phi0=5,
                       mechanisms=c("MCAR","MAR","MNAR"), miss_pcts=c(15,25,35), methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),test_dir="",
                       outfile=NULL){
  list_res=list(); index=1
  for(s in 1:length(sim_index)){
    for(d in 1:length(datasets)){
      dataset=datasets[d]
      dir_name=sprintf("Results/%s/phi%d/sim%d%s",dataset,phi0,sim_index[s],test_dir)
      data_dir_name=sprintf("Results/%s/phi%d/sim%d",dataset,phi0,sim_index[s])
      imputation_metrics=c("MSE","L1","L2","NRMSE","RMSE")
      for(i in 1:length(imputation_metrics)){
        res = processComparisons(dir_name,data_dir_name,mechanisms,miss_pcts,methods,imputation_metrics[i],
                                      betaVAE=F, arch="IWAE", rdeponz=F, outfile=outfile)
        list_res[[index]] = res
        index = index+1
      }
    }
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


summarize_Physionet = function(prefix="",mechanism="MNAR",miss_pct=NA,dataset="Physionet"){
  dir_name = sprintf("Results/%s/phi5/sim1/%s",dataset,prefix)
  load(sprintf("%s/data_%s_%d.RData",dir_name,mechanism,miss_pct))
  library(ggplot2)
  library(ggfortify)
  library(gridExtra)
  # load and extract info here
  d1 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt")
  d2 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-b.txt")
  d3 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-c.txt")
  classes = c(d1$In.hospital_death, d2$In.hospital_death, d3$In.hospital_death)    # mortality

  datas = split(data.frame(data), g)        # split by $train, $test, and $valid
  datas$test = data

  Missings = split(data.frame(Missing), g)
  Missings$test = Missing

  norm_means=colMeans(datas$train); norm_sds=apply(datas$train,2,sd)

  load(sprintf("%s/res_NIMIWAE_%s_%d_IWAE_rzF_ignorable.RData",dir_name,mechanism,miss_pct))
  res0 = res_NIMIWAE
  xhat0 = res_NIMIWAE$xhat_rev

  load(sprintf("%s/res_NIMIWAE_%s_%d_IWAE_rzF.RData",dir_name,mechanism,miss_pct))
  res1= res_NIMIWAE
  xhat1 = res_NIMIWAE$xhat_rev

  load(sprintf("%s/res_MIWAE_%s_%d.RData",dir_name,mechanism,miss_pct))
  xhat2 = res_MIWAE$xhat_rev

  load(sprintf("%s/res_HIVAE_%s_%d.RData",dir_name,mechanism,miss_pct))
  xhat3 = res_HIVAE$data_reconstructed

  load(sprintf("%s/res_VAEAC_%s_%d.RData",dir_name,mechanism,miss_pct))
  xhat_all = res_VAEAC$result
  xhat4 = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
  n_imputations = res_VAEAC$train_params$n_imputations
  for(i in 1:nrow(datas$test)){
    xhat4[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),])
  }

  load(sprintf("%s/res_MEAN_%s_%d.RData",dir_name,mechanism,miss_pct))
  xhat5 = res_MEAN$xhat_rev

  load(sprintf("%s/res_MF_%s_%d.RData",dir_name,mechanism,miss_pct))
  xhat6 = res_MF$xhat_rev

  # 0: NIMIWAE_Ignorable, 1: NIMIWAE, 2: MIWAE, 3: HIVAE
  # 4: VAEAC, 5: MEAN, 6: MF
  for(ii in 1:ncol(xhat0)){
    idy=ii
    if(all(Missing[,idy]==1)){next}

    png(sprintf("%s/Diagnostics/MEAN_col%d.png",
                dir_name,idy))
    boxplot(xhat0[Missing[,idy]==0,idy],xhat1[Missing[,idy]==0,idy],xhat2[Missing[,idy]==0,idy],xhat3[Missing[,idy]==0,idy],xhat4[Missing[,idy]==0,idy],
            xhat6[Missing[,idy]==0,idy],
            names=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MissForest"),
            main=sprintf("Boxplot of Imputed Values of %s",colnames(data)[idy]),outline=F,las=2)
    abline(h=xhat5[Missing[,idy]==0,idy][1],col="red"); dev.off()

    df = data.frame(method=rep(c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MissForest"),each=sum(Missing[,idy]==0)),
                    L1=c(xhat0[Missing[,idy]==0,idy],xhat1[Missing[,idy]==0,idy],xhat2[Missing[,idy]==0,idy],xhat3[Missing[,idy]==0,idy],xhat4[Missing[,idy]==0,idy],
                         xhat6[Missing[,idy]==0,idy])  )
    p=ggplot(df,aes(x=method,y=value)) +
      geom_boxplot(out)

    imputed_data = cbind(xhat0[Missing[,idy]==0,idy],
                         xhat1[Missing[,idy]==0,idy],
                         xhat2[Missing[,idy]==0,idy],
                         xhat3[Missing[,idy]==0,idy],
                         xhat4[Missing[,idy]==0,idy],
                         xhat5[Missing[,idy]==0,idy],
                         xhat6[Missing[,idy]==0,idy])
    colnames(imputed_data) = c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF")
    imputed_data = as.data.frame(imputed_data)

    p1 = ggplot(imputed_data, aes(x=VAEAC, y=NIMIWAE)) + geom_point() + geom_abline(slope=1,intercept=0,col="red")
    ggsave(filename=sprintf("%s/Diagnostics/NIMvVAEAC_col%d.png",
                            dir_name,idy),plot=p1)

    p2 = ggplot(imputed_data, aes(x=IMIWAE, y=NIMIWAE)) + geom_point() + geom_abline(slope=1,intercept=0,col="red")
    ggsave(filename=sprintf("%s/Diagnostics/NIMvIM_col%d.png",
                            dir_name,idy),plot=p2)

    p3 = ggplot(imputed_data, aes(x=IMIWAE, y=VAEAC)) + geom_point() + geom_abline(slope=1,intercept=0,col="red")
    ggsave(filename=sprintf("%s/Diagnostics/VAEACvIM_col%d.png",
                            dir_name,idy),plot=p3)


  }

  # # idy=14 # 12 (MAP) and 14 (NIMAP)
  # idy = 13 # 13 (DBP), 14 (MAP), 15 (SBP)
  # # 19 (ALP) and 23 (BUN)

  miss_given_mortality = matrix(nrow=3,ncol=2)    ### LOOKING AT DBP, MAP, SBP, respectively
  for(ii in 13:15){
    idy=ii
    tab = table(classes,Missing[,idy])
    # apply(tab,1,function(x) (x/sum(x)))   ## look down column: given mortality =0 or 1, % observed for variable
    # apply(tab,2,function(x) (x/sum(x)))   ## look down column: given missing in variable, % mortality
    miss_given_mortality[ii-12,] = apply(tab,1,function(x) (x/sum(x)))[2,]   ## Of patients who survived (0) or died (1), proportion of that feature that was observed
  }
  miss_given_mortality = 1-miss_given_mortality
  colnames(miss_given_mortality) = c("Survived","Died")
  rownames(miss_given_mortality) = c("Missing DBP","Missing MAP","Missing SBP")
  write.table(miss_given_mortality,file=sprintf("%s/Diagnostics/miss_BPs.txt",dir_name))
  png(sprintf("%s/Diagnostics/miss_BPs.png",dir_name), height=500, width=500)
  p<-tableGrob(round(miss_given_mortality,4))
  grid.arrange(p)
  dev.off()


  #########################################
  ############################################
  ################################################

  all_coefs = matrix(NA,ncol=ncol(data),nrow=6)
  all_SEs = matrix(NA,ncol=ncol(data),nrow=6)
  all_Zs = matrix(NA,ncol=ncol(data),nrow=6)
  all_pvals = matrix(NA,ncol=ncol(data),nrow=6)
  all_95Ls = matrix(NA,ncol=ncol(data),nrow=6)
  all_95Hs = matrix(NA,ncol=ncol(data),nrow=6)
  rownames(all_coefs) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  rownames(all_SEs) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  rownames(all_Zs) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  rownames(all_pvals) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  rownames(all_95Ls) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  rownames(all_95Hs) = c("IMIWAE","NIMIWAE","HIVAE","VAEAC","Mean","MF")
  colnames(all_coefs) = c(colnames(data))
  colnames(all_SEs) = c(colnames(data))
  colnames(all_Zs) = c(colnames(data))
  colnames(all_pvals) = c(colnames(data))
  colnames(all_95Ls) = c(colnames(data))
  colnames(all_95Hs) = c(colnames(data))

  for(j in 1:6){
    xhat = if(j==1){ xhat0  # IMIWAE
    }else if(j==2){ xhat1  # NIMIWAE
    }else if(j==3){ xhat=xhat3  # HIVAE
    }else if(j==4){ xhat=xhat4  # VAEAC
    }else if(j==5){ xhat=xhat5  # Mean
    }else if(j==6){ xhat=xhat6  # MF
    }
    # load and extract info here
    d1 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt")
    d2 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-b.txt")
    d3 = read.csv("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-c.txt")

    classes_test = c(d1$In.hospital_death, d2$In.hospital_death, d3$In.hospital_death)    # mortality


    train_ids = which(g%in%c("train","valid")); test_ids = which(g=="test"); xhat2=xhat; classes_test2=classes_test
    classes_test=classes_test2

    x = data.frame(cbind(xhat[train_ids,]))

    # library(ggplot2)
    library(gridExtra)

    if(resp=="mortality"){
      x$class = as.factor(classes_test[train_ids])
      model <- glm(class ~ 0 + ., data=x, family = "binomial")
      coefs = model$coefficients
      coefs_x = coefs[order(-coefs)]
      names_x = names(data)[order(-coefs)]

      tab=cbind(c(names_x[1:10]),round(c(coefs_x[1:10]),3))
      p<-tableGrob(tab)
      ggsave(p,file=sprintf("Results/%s%s/phi5/sim1/Diagnostics/%s_LR_coefs.png",prefix,dataset,method),
               width=6, height=6)
    }

    # FOR LR ONLY
    all_coefs[j,] = summary(model)$coefficients[,1]
    all_SEs[j,] = summary(model)$coefficients[,2]
    all_Zs[j,] = summary(model)$coefficients[,3]
    all_pvals[j,] = summary(model)$coefficients[,4]
    all_95Ls[j,] = summary(model)$coefficients[,1] - 1.96*summary(model)$coefficients[,2]
    all_95Hs[j,] = summary(model)$coefficients[,1] + 1.96*summary(model)$coefficients[,2]
  }

  ordered_coefs = t(all_coefs[,order(-abs(all_coefs[2,]))])
  ordered_SEs = t(all_SEs[,order(-abs(all_coefs[2,]))])
  ordered_pvals = t(all_pvals[,order(-abs(all_coefs[2,]))])

  head(ordered_coefs,n=10)
  head(ordered_SEs,n=10)
  tail(ordered_coefs,n=10)

  LR_coefs=cbind(rep(colnames(all_coefs), each=length(rownames(all_coefs))),
                 rep(rownames(all_coefs), times=length(colnames(all_coefs))),
                 c(round(all_coefs,2)),
                 c(round(all_SEs,2)),
                 c(round(all_Zs,3)),
                 c(round(all_pvals,2)),
                 paste("(",c(round(all_95Ls,2)),",",c(round(all_95Hs,2)),")",sep=""))
  colnames(LR_coefs) = c("Effect","Method","Estimate","SE","Z-stat","p value","95% Interval")

  top5 = c("pH_last","FiO2_last","MechVentLast8Hour","Mg_last","Lactate_last")
  LR_coefs_top5 = LR_coefs[LR_coefs[,1] %in% top5,]
  LR_coefs_top5[c(1:nrow(LR_coefs_top5))%%6!=1,1] = ""
  # LR_coefs_top5[LR_coefs_top5[,2]=="NIMIWAE0",2]="IMIWAE"


  save(list=c("LR_coefs","LR_coefs_top5"),file=sprintf("%s/Diagnostics/LR_coefs.RData",dir_name))

  png(sprintf("%s/Diagnostics/LR_coefs.png",dir_name), height=2800, width=2100,res=300)
  p<-tableGrob(LR_coefs_top5)
  grid.arrange(p)
  dev.off()


}
