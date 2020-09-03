library(reticulate)
library(NIMIWAE)
runComparisons = function(mechanism=c("MCAR","MAR","MNAR"), miss_pct=25, miss_cols=NULL, ref_cols=NULL, scheme="UV",
                          sim_params=list(N=1e5, D=1, P=2, seed = NULL),
                          dataset=c("Physionet_mean","Physionet_all","HEPMASS","POWER","GAS","IRIS","RED","WHITE","YEAST","BREAST","CONCRETE","BANKNOTE",
                                    "SIM"),
                          save.folder=dataset, save.dir=".",
                          run_methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi_0=5,sim_index=1){
  np = import('numpy',convert=FALSE)
  source("processComparisons.R")

  ## Simulate data ##
  if(!is.null(sim_params$seed)){sim_params$seed = sim_params$sim_index*9} # default seed for reproducibility

  dir_name1=sprintf("%s/Results",save.dir)
  dir_name=sprintf("%s/Results/%s/phi%d/sim%d",save.dir,save.folder,phi_0,sim_index)      # this directory is where everything will be saved

  ifelse(!dir.exists(dir_name),dir.create(dir_name,recursive=T),FALSE)
  fname_data=sprintf("%s/data_%s_%d",dir_name,mechanism,miss_pct)

  # Simulate data if data file doesn't exist. Otherwise, load the data file
  #### REPLACE WITH simulate_data(), read_data() and simulate_missing()
  if(!file.exists(sprintf("%s.RData",fname_data))){
    print("Preparing data")
    if(dataset=="SIM"){ fit_data = NIMIWAE::simulate_data( sim_params$N, sim_params$D, sim_params$P, sim_index, seed = 9*sim_index, ratio=c(6,2,2) )
    } else{ fit_data = NIMIWAE::read_data( dataset=dataset, ratio=c(6,2,2) ); data=fit_data$data; classes=fit_data$classes }
    n=nrow(data); p=ncol(data)

    # default phi=5
    set.seed(222); phis=rlnorm(p,log(phi_0),0.2)  # draw phi from log-normal with log-mean phi_0, sd=0.2
    phi_z=NULL  # dependence on Z (latent variable): disabled for fmodel="S" (Selection model). Used for fmodel="PM" (Pattern-mixture model)

    ## Simulate Missing ##
    if(!grepl("Physionet",dataset)){
      print(sprintf("Simulating %s missingness",mechanism))
      Missing=matrix(1L,nrow=nrow(data),ncol=ncol(data))    # all observed unless otherwise specified
      if(is.null(miss_cols)){
        set.seed(111)   # random selection of anchors/missing features
        ref_cols=sample(c(1:ncol(data)),ceiling(ncol(data)/2),replace=F)    # more anchors than missing --> true miss_pct always < miss_pct
        miss_cols=(1:ncol(data))[-ref_cols]
      }
      print(paste("ref_cols:",paste(ref_cols,collapse=",")))
      print(paste("miss_cols:",paste(miss_cols,collapse=",")))

      # weight miss_pct to simulate appropriate amount of missing overall (in entire dataset)
      weight=p/length(miss_cols)
      miss_pct2=miss_pct*weight
      pi=1-miss_pct2/100
      fit_missing=simulate_missing(data.matrix(data), miss_cols, ref_cols, pi,
                                   phis, phi_z,
                                   scheme, mechanism, sim_index, fmodel="S") # Z is NULL for Selection model (fmodel="S"). Z required for Pattern-mixture model (fmodel="PM")
      Missing=fit_missing$Missing; prob_Missing=fit_missing$probs      # missing mask, probability of each observation being missing unknown
    } else{  # Physionet_mean and Physionet_all have inherent missingness, no missingness simulated. Code "Missing" to be consistent with notation
      library(reticulate)
      np <- import("numpy")
      npz1 <- np$load("data/PhysioNet2012/physionet.npz")

      if(strsplit(dataset,"_")[[1]][2] == "mean"){
        # dataset=="Physionet_mean"
        Missing = 1 - floor(rbind(apply(npz1$f$m_train_miss, c(1,3), mean),
                                  apply(npz1$f$m_val_miss, c(1,3), mean),
                                  apply(npz1$f$m_test_miss, c(1,3), mean)))
      } else if(strsplit(dataset,"_")[[1]][2] == "all"){
        # dataset=="Physionet_all"
        library(abind)
        M3D = aperm(abind(npz1$f$m_train_miss,
                          npz1$f$m_val_miss,
                          npz1$f$m_test_miss,
                          along=1),
                    c(2,1,3))                                             # switch dims so 48 time points is first dim
        Missing = 1 - matrix(M3D, nrow=dim(M3D)[1]*dim(M3D)[2], ncol=dim(M3D)[3])   # stack time series data: 1st subject is 1st - 48th observations, 2nd subj is 49th - 96th, ...
      }
      fit_missing=NULL; prob_Missing=Missing; ref_cols=NULL; miss_cols=NULL
    }

    ## Create Validation/Training/Test splits here. default: 60%-20%-20% ##
    # ratios=c(train = .6, test = .2, valid = .2)
    #
    # set.seed(333)
    # g = sample(cut(
    #   seq(nrow(data)),
    #   nrow(data)*cumsum(c(0,ratios)),
    #   labels = names(ratios)
    # ))

    g = fit_data$g
    save(list=c("data","Missing","fit_data","fit_missing","prob_Missing","g","ref_cols","miss_cols"),file=sprintf("%s.RData",fname_data))
  }else{
    print("Loading previously simulated data")
    load(sprintf("%s.RData",fname_data))
  }

  # Plotting diagnostics
  # if(trace){
  #   diag_dir_name = sprintf("%s/Diagnostics/miss%d",dir_name,miss_pct)
  #   ifelse(!dir.exists(diag_dir_name),dir.create(diag_dir_name,recursive=T),F)
  #   for(c in miss_cols){
  #     png(sprintf("%s/%s_col%d_Truth_allData.png",diag_dir_name,mechanism,c))
  #     p = ggplot(data.frame(value=data[,c]),aes(value)) + geom_density(alpha=0.2) + ggtitle(sprintf("Column %d: Density plot of all observations",c))
  #     print(p)
  #     dev.off()
  #     png(sprintf("%s/%s_col%d_Truth_MissVsObs.png",diag_dir_name,mechanism,c))
  #     overlap_hists(x1=data[Missing[,c]==0,c],lab1="Missing",
  #                   x2=data[Missing[,c]==1,c],lab2="Observed",
  #                   title=sprintf("Column %d: Density plot of Missing vs Observed Observations",c))
  #     dev.off()
  #   }
  # }



  #######################
  ### RUN EACH METHOD ###
  #######################
  # Train on training set, validate hyperparameters on validation set, and impute test set
  # Imputation metrics based on imputed test set only
  # Skips method if it isn't included in "run_methods"

  #### NIMIWAE ####
  print("NIMIWAE")
  if("NIMIWAE" %in% run_methods){
    # Fixed:
    dec_distrib="Normal"; learn_r=T

    covars_r=rep(1,ncol(data)); ignorable=F  # all as covariates
    # if(mechanism=="MCAR"){ignorable=T; covars_r = rep(0,ncol(data)); print(covars_r)
    # }else if(mechanism=="MAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[ref_cols] = 1; print(covars_r)  # ignorable, but testing nonignorable with MAR --> should still be fine
    # } else if(mechanism=="MNAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[miss_cols] = 1; print(covars_r)}

    # Variants:
    # if(dataset%in%c("TOYZ","TOYZ2")){rdeponzs = c(F,T); archs = c("IWAE","VAE"); betaVAEs = c(F,T)
    # }else{
    rdeponz = c(F); arch = c("IWAE"); betaVAE = c(F)
    # }
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    yesbeta=if(betaVAE){"beta"}else{""}; yesrz = if(rdeponz){"T"}else{"F"}
    fname0=sprintf("%s/res_NIMIWAE_%s_%d_%s%s_rz%s.RData",dir_name2,mechanism,miss_pct,yesbeta,arch,yesrz)

    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_NIMIWAE = tuneHyperparams(method="NIMIWAE",data=data,Missing=Missing,g=g,
                                    rdeponz=rdeponz, learn_r=learn_r,
                                    phi0=phi0,phi=phi,
                                    covars_r=covars_r, dec_distrib=dec_distrib,
                                    arch=arch, betaVAE=betaVAE, ignorable=ignorable)

      res_NIMIWAE$time = as.numeric(Sys.time()-t0,units="secs")
      print(paste("Time elapsed: ", res_NIMIWAE$time, "s."))
      save(res_NIMIWAE,file=sprintf("%s",fname0))
    }else{
      load(sprintf("%s",fname0))   # if results already exist, load to plot diagnostic plots
    }
    # Diagnostic plots of results (imputed vs truth)
    for(c in miss_cols){
      png(sprintf("%s/%s_col%d_NIMIWAE_%s%s_rz%s.png",diag_dir_name,mechanism,c,yesbeta,archs[aa],yesrz))
      overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                    x2=res_NIMIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                    x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                    title=sprintf("NIMIWAE Column%d: True vs Imputed missing and observed values",c))
      dev.off()
    }

    rm("res_NIMIWAE")  # if running multiple methods, save memory by saving results and removing from environment
  }

  #### MIWAE  (baseline comparison) ####
  print("MIWAE")
  if("MIWAE" %in% run_methods){
    rdeponz=FALSE;  covars_r=NULL; dec_distrib="Normal"; learn_r=NULL
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MIWAE_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MIWAE = tuneHyperparams(method="MIWAE",data=data,Missing=Missing,g=g,
                                  rdeponz=rdeponz, learn_r=learn_r,
                                  phi0=phi0,phi=phi,
                                  covars_r=covars_r, dec_distrib=dec_distrib)
      res_MIWAE$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MIWAE,file=sprintf("%s",fname0))
      #rm("res_MIWAE")
    }else{
      load(fname0)
      for(c in miss_cols){
        png(sprintf("%s/%s_col%d_MIWAE.png",diag_dir_name,mechanism,c))
        overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                      x2=res_MIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                      x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                      title=sprintf("MIWAE Column%d: True vs Imputed missing and observed values",c))
        dev.off()
      }
    }
    rm("res_MIWAE")
  }

  #### SETUP: Other methods ####
  data_types=list()
  one_hot_max_sizes=rep(NA,ncol(data))
  for(i in 1:ncol(data)){
    # factors/ordinal --> just treat as categorical to automatize
    if(is.character(data[,i]) | is.factor(data[,i])){
      nclass=as.character('length(unique(data[,i]))')
      data_types[i]=list(type='cat',dim=nclass,nclass=nclass)

      one_hot_max_sizes[i]=as.integer(nclass)
    }
    # numeric (real/pos/count)
    if(is.numeric(data[,i])){
      # positive
      if(all(data[,i]>=0)){
        # count (count is positive)
        if(all(data[,i]==round(data[,i],0))){
          data_types[[i]]=list(type='count',dim='1',nclass='')
        } else{
          data_types[[i]]=list(type='pos',dim='1',nclass='')
        }
      } else{
        data_types[[i]]=list(type='real',dim='1',nclass='')
      }

      one_hot_max_sizes[i]=1L
    }
  }

  # for VAEAC
  MissingData = data
  MissingData[Missing==0]=NaN

  MissingDatas = split(data.frame(MissingData),g)
  source_python("comparisons_missing.py")

  ### HIVAE ###
  print("HIVAE")
  if("HIVAE" %in% run_methods){
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_HIVAE_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_HIVAE = tuneHyperparams(method="HIVAE",data=data,Missing=Missing,g=g,
                                  data_types=data_types)
      res_HIVAE$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_HIVAE,file=sprintf("%s",fname0))
      #rm("res_HIVAE")
    }else{
      load(fname0)
      for(c in miss_cols){
        png(sprintf("%s/%s_col%d_HIVAE.png",diag_dir_name,mechanism,c))
        overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                      x2=res_HIVAE$data_reconstructed[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                      x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                      title=sprintf("HIVAE Column%d: True vs Imputed missing and observed values",c))
        dev.off()
      }
    }
    rm("res_HIVAE")
  }

  ### VAEAC ###
  print("VAEAC")
  if("VAEAC" %in% run_methods){
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_VAEAC_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    # n_hidden_layers = if(dataset%in% c("TOYZ","TOYZ2","BANKNOTE","IRIS","WINE","BREAST","YEAST","CONCRETE","SPAM","ADULT","GAS","POWER")){10L} # default
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_VAEAC = tuneHyperparams(method="VAEAC",data=data,Missing=Missing,g=g,
                                  one_hot_max_sizes=one_hot_max_sizes, MissingDatas=MissingDatas)
      res_VAEAC$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_VAEAC,file=sprintf("%s",fname0))
      #rm("res_VAEAC")
    }else{
      load(fname0)
      xhat_all = res_VAEAC$result    # this method reverses normalization intrinsically
      # average imputations
      xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
      n_imputations = res_VAEAC$train_params$n_imputations
      for(i in 1:nrow(datas$test)){ xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),]) }

      for(c in miss_cols){
        png(sprintf("%s/%s_col%d_VAEAC.png",diag_dir_name,mechanism,c))
        overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                      x2=xhat[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                      x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                      title=sprintf("VAEAC Column%d: True vs Imputed missing and observed values",c))
        dev.off()
      }
    }
    rm("res_VAEAC")
  }

  ### MF ###
  print("MF")
  if("MF" %in% run_methods){
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MF_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MF = tuneHyperparams(method="MF",data=data,Missing=Missing,g=g)
      res_MF$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MF,file=sprintf("%s",fname0))
      #rm("res_MF")
    }else{
      load(fname0)
      if(is.null(res_MF$xhat_rev)){res_MF$xhat_rev = reverse_norm_MIWAE(res_MF$xhat_mf,norm_means,norm_sds)}
      for(c in miss_cols){
        png(sprintf("%s/%s_col%d_MF.png",diag_dir_name,mechanism,c))
        overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing",
                      x2=res_MF$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                      x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                      title=sprintf("MF Column%d: True vs Imputed missing and observed values",c))
        dev.off()
      }
    }
    rm("res_MF")
  }

  ### MEAN ###
  print("MEAN")
  if("MEAN" %in% run_methods){
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MEAN_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MEAN = tuneHyperparams(method="MEAN",data=data,Missing=Missing,g=g)
      res_MEAN$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MEAN,file=sprintf("%s",fname0))
      #rm("res_MEAN")
    }else{
      load(fname0)
      if(is.null(res_MEAN$xhat_rev)){res_MEAN$xhat_rev = reverse_norm_MIWAE(res_MEAN$xhat_mean,norm_means,norm_sds)}
      for(c in miss_cols){
        png(sprintf("%s/%s_col%d_MEAN.png",diag_dir_name,mechanism,c))
        overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
                      x2=res_MEAN$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
                      x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
                      title=sprintf("MEAN Column%d: True vs Imputed missing and observed values",c))
        dev.off()
      }
    }
    rm("res_MEAN")
  }
}

ifelse(!dir.exists("./Results"), dir.create("./Results",recursive=T), F)

# Proof of Concept (fixed miss_pct)
mechanisms=c("MCAR","MAR","MNAR")
for(a in 1:length(mechanisms)){
  runComparisons(dataset="SIM", sim_params=list(N=1e5, D=1, P=2, seed=NULL), save.dir="./Results", save.folder="SIM1", mechanism=mechanisms[a])
}

# Main simulations (P=8, 5 sims, vary miss_pct)
sim_indexes=1:5; miss_pcts = c(15,25,35)
for(a in 1:length(mechanisms)){for(b in 1:length(miss_pcts)){for(c in 1:length(sim_indexes)){
  runComparisons(dataset="SIM", sim_params=list(N=1e5, D=2, P=8, seed=NULL), save.dir="./Results",save.folder="SIM2", mechanism=mechanisms[a],miss_pct=miss_pcts[b], sim_index=sim_indexes[c])
}}}

# UCI datasets (fixed miss_pct)
datasets=c("BANKNOTE","CONCRETE","RED","WHITE")
for(a in 1:length(mechanisms)){for(d in 1:length(datasets)){
  runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a])
}}
datasets=c("HEPMASS","POWER") # large datasets: runs into memory issues with missForest (MF)
for(a in 1:length(mechanisms)){for(d in 1:length(datasets)){
  runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a], run_methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN"))
}}

# Physionet analysis
runComparisons(dataset="Physionet_mean", save.dir="./Results",mechanism="MNAR")  # for Physionet, missingness isn't simulated. Inherent missingness is assumed to be MNAR
