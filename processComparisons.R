reverse_norm_MIWAE = function(x,norm_means,norm_sds){
  xnew=matrix(nrow=nrow(x),ncol=ncol(x))
  for(i in 1:ncol(xnew)){
    xnew[,i]=(x[,i]*(norm_sds[i]))+norm_means[i]
  }
  return(xnew)
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

output_file.name=function(dir_name,method=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),
                          mechanism,miss_pct,betaVAE,arch,rdeponz){
  if(method=="NIMIWAE"){
    yesbeta = if(betaVAE){"beta"}else{""}; yesrdeponz = if(rdeponz){"rzT"}else{"rzF"}
    file.name=sprintf("%s/res_%s_%s_%d_%s%s_%s",
                      dir_name,method,mechanism,miss_pct,yesbeta,arch,yesrdeponz)
  }else{
    file.name=sprintf("%s/res_%s_%s_%d",
                      dir_name,method,mechanism,miss_pct) # for miwae, default = Normal (StudentT can be done later)
  }
  print(file.name)
  file.name=sprintf("%s.RData",file.name)
  return(file.name)
}

processComparisons=function(dir_name="Results/SIM1/phi5",mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(15,25,35),
                            methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),
                            imputation_metric=c("MSE","NRMSE","L1","L2"), betaVAE=NULL, arch=NULL, rdeponz=NULL, outfile=NULL){
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
                                   miss_pct=miss_pct[ii], betaVAE=betaVAE, arch=arch, rdeponz=rdeponz)
      list_res[[index]]=toy_process(data.file.name,file.name,methods[j])
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
                       mechanisms=c("MCAR","MAR","MNAR"), miss_pcts=c(15,25,35), methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),test_dir="",
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
                      methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi0=5,sim_index=1,
                      betaVAE=NULL, arch=NULL, rdeponz=NULL){
  # return df with columns: dataset, method, mechanism, and value (NRMSE, L1, L2, MSE, or RMSE. choose here)
  nvals = length(datasets)*length(mechanisms)*length(miss_pct)*length(methods)*length(sim_index)
  df=data.frame(matrix(nrow=nvals,ncol=10))
  index=1
  colnames(df) = c("dataset","mechanism","method","miss_pct","sim_index","MSE","RMSE","NRMSE","L1","L2")
  for(i in 1:length(datasets)){for(j in 1:length(mechanisms)){for(k in 1:length(methods)){for(l in 1:length(miss_pct)){for(m in 1:length(sim_index)){
    ### LOAD Results/datasets[i]/phi5/sim(index)/res_(method)_(mechanism)_(miss_pct).RData
    if(methods[k]=="NIMIWAE2"){subdir="/Ignorable"; method0="NIMIWAE"}else{subdir=""; method0=methods[k]}
    dir_name0 = sprintf("Results/%s/phi%d/sim%d",datasets[i],phi0,sim_index[m])
    dir_name = sprintf("%s%s",dir_name0,subdir)
    data.file.name=sprintf("%s/data_%s_%d.RData",dir_name0,mechanisms[j],miss_pct[l])
    file.name = output_file.name(dir_name=dir_name,method=method0, mechanism=mechanisms[j],
                                 miss_pct=miss_pct[l], betaVAE=F, arch="IWAE", rdeponz=F)
    df[index,1:5] = c(datasets[i],mechanisms[j],methods[k],miss_pct[l],sim_index[m])
    print(data.file.name)
    print(file.name)
    if(file.exists(file.name)){
      res = toy_process(data.file.name,file.name,method0)$results   # returns unlist(list(MSE=MSE,RMSE=RMSE,NRMSE=NRMSE,L1=L1,L2=L2))
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
