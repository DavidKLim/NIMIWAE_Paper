library(reticulate)
library(NIMIWAE)
source_python("otherMethods.py")

runComparisons = function(mechanism=c("MCAR","MAR","MNAR"), miss_pct=25, miss_cols=NULL, ref_cols=NULL, scheme="UV",
                          sim_params=list(N=1e5, D=1, P=2, seed = NULL),
                          dataset=c("Physionet","HEPMASS","POWER","GAS","IRIS","RED","WHITE","YEAST","BREAST","CONCRETE","BANKNOTE",
                                    "SIM"),
                          save.folder=dataset, save.dir=".",
                          run_method=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE"),phi_0=5,sim_index=1,
                          rdeponz = c(F), arch = c("IWAE"), ignorable=F, n_imputations=25){
  np = import('numpy',convert=FALSE)
  source("processComparisons.R")

  ## Simulate data ##
  if(!is.null(sim_params$seed)){sim_params$seed = sim_params$sim_index*9} # default seed for reproducibility

  dir_name1=sprintf("%s",save.dir)
  dir_name=sprintf("%s/%s/phi%d/sim%d",save.dir,save.folder,phi_0,sim_index)      # this directory is where everything will be saved

  ifelse(!dir.exists(dir_name),dir.create(dir_name,recursive=T),FALSE)
  fname_data=sprintf("%s/data_%s_%d",dir_name,mechanism,miss_pct)

  # Simulate data if data file doesn't exist. Otherwise, load the data file
  #### REPLACE WITH simulate_data(), read_data() and simulate_missing()
  if(!file.exists(sprintf("%s.RData",fname_data))){
    print("Preparing data")

    n = sim_params$N; p = sim_params$P

    miss_cols=NULL;ref_cols=NULL;phis=NULL;phi_z=NULL
    if(grepl("NONLINEAR",toupper(dataset))){
      scheme="NL"; nonlinear=T
    } else{ scheme="UV"; nonlinear=F }

    if(grepl("SIM",dataset)){
      fit_data = NIMIWAE::simulate_data( N=sim_params$N, D=sim_params$D, P=sim_params$P, sim_index=sim_index, seed = 9*sim_index, ratio=c(8,2), nonlinear=nonlinear )
      data=fit_data$data; classes=fit_data$classes
    } else if(grepl("Physionet",dataset)){ fit_data = NIMIWAE::read_data( dataset=dataset, ratio=c(8,2) ); data=fit_data$data; classes=fit_data$classes }
    if(is.null(phis)){set.seed(222); phis=rlnorm(p,log(phi_0),0.2)}else if(length(phis)==1){phis=rep(phis,p)}
    if(is.null(phi_z)){phi_z=phis[1]/length(unique(classes))} # set dependence on class as phi/#classes (deprecated.)

    ## Simulate Missing ##
    if(!grepl("Physionet",dataset)){
      # default phi=5
      set.seed(222); phis=rlnorm(p,log(phi_0),0.2)  # draw phi from log-normal with log-mean phi_0, sd=0.2
      phi_z=NULL  # dependence on Z (latent variable): disabled for fmodel="S" (Selection model). Used for fmodel="PM" (Pattern-mixture model)

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
      g = fit_data$g

      phi0=fit_missing$params[[1]]$phi0; phi=fit_missing$params[[1]]$phi

    } else{  # Physionet_mean and Physionet_all have inherent missingness, no missingness simulated. Code "Missing" to be consistent with notation
      library(reticulate)
      # np <- import("numpy")
      # npz1 <- np$load("data/PhysioNet2012/physionet.npz")
      if(!dir.exists("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0")){
        # if training set doesn't exist, assume it hasn't been downloaded
        setwd("data")

        print("Downloading Physionet 2012 Challenge Dataset...")
        download.file("https://physionet.org/static/published-projects/challenge-2012/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0.zip",
                      destfile="predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0.zip")

        print("Unzipping compressed directory...")
        unzip("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0.zip")

        print("Removing zip file...")
        file.remove("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0.zip")

        if(!file.exists("set_c_merged.h5")){
          print("Merging into one dataset for summary table (Table 1)...")
          source_python("raw_data_gather.py")
        }

        setwd("predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0")

        ###########################################################################
        # Creating Table 1 (done first time runComparisons.R is run)
        library(reticulate)
        np <- import("numpy")

        features = c('ALP','ALT','AST','Albumin','BUN','Bilirubin',
                     'Cholesterol','Creatinine','DiasABP','FiO2','GCS',
                     'Glucose','HCO3','HCT','HR','K','Lactate','MAP', 'MechVent',
                     'Mg','NIDiasABP','NIMAP','NISysABP','Na','PaCO2',
                     'PaO2','Platelets','RespRate','SaO2','SysABP','Temp',
                     'TroponinI','TroponinT','Urine','WBC','pH')

        npz_trainval <- np$load("data_train_val.npz")
        npz_test <- np$load("data_test.npz")
        d_train = npz_trainval$f$x_train; d_val = npz_trainval$f$x_val; d_test = npz_test$f$x_test
        M_train = npz_trainval$f$m_train; M_val = npz_trainval$f$m_val; M_test = npz_test$f$m_test

        library(abind)
        d3=abind(d_train, d_val, d_test, along = 1)
        M3=abind(M_train, M_val, M_test, along = 1)

        d3=aperm(d3, c(2,1,3)); M3=aperm(M3,c(2,1,3))
        d = matrix(d3, nrow=dim(d3)[1]*dim(d3)[2], ncol=dim(d3)[3])
        M = 1-matrix(M3, nrow=dim(M3)[1]*dim(M3)[2], ncol=dim(M3)[3])
        colnames(d) = features; colnames(M) = features

        nobs = rep(NA,ncol(d)); n1obs = rep(NA,ncol(d))
        for(c in 1:ncol(d)){
          nobs[c] = sum(M[,c]==1) #; n1obs[c] = any(M[,c]==1)   # n1obs: number of subjects with at least one non-missing obs for each feature
          nonmiss = rep(NA,nrow(M)/48) # number of obs with at least 1 nonmissing entry for feature c
          for(b in 1:(nrow(M)/48)){
            nonmiss[b] = any(M[(b-1)*48+(1:48) , c]==1)
          }
          n1obs[c] = sum(nonmiss)
        }

        ## for table in data section of P2 paper: feature, %missing, %(subjects with at least one nonmissing entry)
        tab1 = cbind(features,1-nobs/nrow(d),n1obs/(nrow(d)/48))
        colnames(tab1)[2:3] = c("\\% Missingness", "Patients with $\\geq 1$ measurements")
        tab1[,2] = format(as.numeric(tab1[,2]),digits=2); tab1[,3] = format(as.numeric(tab1[,3]),digits=2)

        save(tab1,file="Tab1.out")
        ############################################################################



        ## run alistair et al pre-processing --> break down into first/last/median/...
        print("Processing data...")
        source_python("../alistair_preprocessing.py")

        if(!dir.exists("set-c")){ untar("set-c.tar.gz") }
        if(!file.exists("PhysionetChallenge2012-set-a.csv")){ process_Alistair('set-a') }
        if(!file.exists("PhysionetChallenge2012-set-b.csv")){ process_Alistair('set-b') }
        if(!file.exists("PhysionetChallenge2012-set-c.csv")){ process_Alistair('set-c') }

        ## read-in pre-processed data
        ## filter to just Median or last observed value
        # d1 = read.csv("PhysionetChallenge2012-set-a.csv")
        # d2 = read.csv("PhysionetChallenge2012-set-b.csv")
        # d3 = read.csv("PhysionetChallenge2012-set-c.csv")
        # library(dplyr)
        # features = c('recordid','SAPS.I','SOFA','Length_of_stay','Survival','In.hospital_death',
        #              'Age','Gender','Height','Weight','CCU','CSRU','SICU',
        #              'DiasABP_median','GCS_median','Glucose_median','HR_median','MAP_median','NIDiasABP_median',
        #              'NIMAP_median','NISysABP_median','RespRate_median','SaO2_median','Temp_median',
        #              'ALP_last','ALT_last','AST_last','Albumin_last','BUN_last','Bilirubin_last',
        #              'Cholesterol_last','Creatinine_last','FiO2_last','HCO3_last','HCT_last','K_last',
        #              'Lactate_last','Mg_last','Na_last','PaCO2_last','PaO2_last','Platelets_last',
        #              'SysABP_last','TroponinI_last','TroponinT_last','WBC_last','Weight_last','pH_last',
        #              'MechVentStartTime','MechVentDuration','MechVentLast8Hour','UrineOutputSum')
        #
        # ## save filtered data
        # d1 %>% select(features) %>% write.csv("PhysionetChallenge2012-set-a.csv", row.names=FALSE)
        # d2 %>% select(features) %>% write.csv("PhysionetChallenge2012-set-b.csv", row.names=FALSE)
        # d3 %>% select(features) %>% write.csv("PhysionetChallenge2012-set-c.csv", row.names=FALSE)

      } else{
        setwd("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0")
      }

      d1 = read.csv("PhysionetChallenge2012-set-a.csv")
      d2 = read.csv("PhysionetChallenge2012-set-b.csv")
      d3 = read.csv("PhysionetChallenge2012-set-c.csv")

      classes = c(d1$"In.hospital_death", d2$"In.hospital_death", d3$"In.hospital_death")
      fit_data = list(classes=classes)

      setwd("../..")

      d1 = d1[,-c(2:6)]  # remove outcome variables (survival, mortality indicator, SAPS/SOFA/length of stay)
      d2 = d2[,-c(2:6)]
      d3 = d3[,-c(2:6)]

      data = rbind(d1,d2,d3); data = data[,-1]   # remove recordid

      Missing = 1 - (is.na(data)^2)

      # ntrain = 4000; nvalid = 4000; ntest = 4000
      # ids = c( rep("train", ntrain), rep("valid", nvalid), rep("test", ntest) ); g = ids

      ratio = c(train = 8,valid = 2)
      ratio = ratio/sum(ratio)
      set.seed(333)
      g = sample(cut(
        seq(nrow(data)),
        nrow(data)*cumsum(c(0,ratio)),
        labels = c("train","valid")
      ))

      fit_missing=NULL; prob_Missing=Missing; ref_cols=NULL; miss_cols=NULL; phi0=NULL; phi=NULL
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

    P = ncol(data)
    data_types=rep("real",P)
    if(grepl("_sup",run_method)){
      ## supervised methods
      print("Supervised method. Concatenating classes")
      data = cbind(data, fit_data$classes)
      Missing = cbind(Missing, rep(1, nrow(Missing)))  # include 1's for classes (fully observed)
      data_types=c(data_types,"cat")
      print("dim(data):")
      print(dim(data))
    }

    save(list=c("data","data_types","Missing","fit_data","fit_missing","prob_Missing","g","ref_cols","miss_cols"),file=sprintf("%s.RData",fname_data))
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

  diag_dir_name = sprintf("%s/Diagnostics",dir_name)
  ifelse(!dir.exists(diag_dir_name),dir.create(diag_dir_name),FALSE)
  #### NIMIWAE ####
  if(grepl("^NIMIWAE",run_method)){
    print(run_method)
    covars_r=rep(1,ncol(data))
    # Fixed:
    dec_distrib="Normal"; learn_r=T

    # if(mechanism=="MCAR"){ignorable=T; covars_r = rep(0,ncol(data)); print(covars_r)
    # }else if(mechanism=="MAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[ref_cols] = 1; print(covars_r)  # ignorable, but testing nonignorable with MAR --> should still be fine
    # } else if(mechanism=="MNAR"){ignorable=F; covars_r = rep(0,ncol(data)); covars_r[miss_cols] = 1; print(covars_r)}

    # Variants:
    # if(dataset%in%c("TOYZ","TOYZ2")){rdeponzs = c(F,T); archs = c("IWAE","VAE"); betaVAEs = c(F,T)
    # }else{
    # }
    # dir_name2=sprintf("%s",dir_name)
    # ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)

    dir_name2=if(ignorable){ sprintf("%s/Ignorable",dir_name) }else{ sprintf("%s",dir_name) }
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)

    if(grepl("_sup", run_method)){ dir_name3 = sprintf("%s/%s_sup/miss%d",dir_name2,mechanism,miss_pct)
    } else{ dir_name3 = sprintf("%s/%s_unsup/miss%d",dir_name2,mechanism,miss_pct) }
    ifelse(!dir.exists(dir_name3),dir.create(dir_name3,recursive=T),FALSE)

    yesrz = if(rdeponz){"T"}else{"F"}

    if(ignorable){
      fname0=sprintf("%s/res_NIMIWAE_%s_%d_%s_rz%s_ignorable.RData",dir_name2,mechanism,miss_pct,arch,yesrz)
    }else{
      fname0=sprintf("%s/res_NIMIWAE_%s_%d_%s_rz%s.RData",dir_name2,mechanism,miss_pct,arch,yesrz)
    }

    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()

      bs=if(grepl("HEPMASS",dataset) | grepl("SIM",dataset)){10000}else if(grepl("BANKNOTE|WINE|BREAST|YEAST|CONCRETE|ADULT|RED",dataset)){200}else if(grepl("POWER",dataset)){2000}else{1000}
      if(grepl("Nonlinear",toupper(dataset))){ n_hidden_layers=c(1L,2L); n_hidden_layers_r0=c(1L,2L,0L)
      }else{ n_hidden_layers=c(0L,1L,2L); n_hidden_layers_r0=c(0L,1L) }
      lr=if(grepl("Nonlinear",toupper(dataset))){ c(0.001, 0.01) }else{ c(0.01,0.001) }; dim_z=as.integer(c(floor(ncol(data)/2),floor(ncol(data)/4)))
      h=c(128L,64L); niw = as.integer(n_imputations*10); n_imputations_per_Z=1L; n_epochs=2002L

      hyperparameters = list(sigma="elu", h=h, n_hidden_layers=n_hidden_layers, n_hidden_layers_r0=n_hidden_layers_r0,
                             bs=bs, lr=lr, dim_z=dim_z, #dim_z=4L, ## or 8, 16, 32?
                             niw = niw, n_imputations=n_imputations_per_Z, n_epochs=n_epochs)

      # res_NIMIWAE = NIMIWAE::NIMIWAE(data=data, dataset=dataset, data_types=data_types, Missing=Missing, g=g, rdeponz=rdeponzs[rr], learn_r=learn_r, phi0=phi0, phi=phi,
      #                                ignorable=ignorable, covars_r=covars_r, arch=archs[aa], draw_xmiss=T,
                                     # hyperparameters=hyperparameters, dir_name=dir_name3, save_imps=save_imps, normalize=normalize)

      save_imps = T; normalize=T; dir_name=dir_name3
      res_NIMIWAE = NIMIWAE::NIMIWAE(dataset=dataset,data=data,Missing=Missing,g=g,
                                    rdeponz=rdeponz, learn_r=learn_r, data_types=data_types,
                                    phi0=phi0,phi=phi, covars_r=covars_r,
                                    arch=arch, ignorable=ignorable,
                                    hyperparameters=hyperparameters, dir_name=dir_name, save_imps=save_imps, normalize=normalize)

      res_NIMIWAE$time = as.numeric(Sys.time()-t0,units="secs")
      print(paste("Time elapsed: ", res_NIMIWAE$time, "s."))
      save(res_NIMIWAE,file=sprintf("%s",fname0))
    }else{
      load(sprintf("%s",fname0))   # if results already exist, load to plot diagnostic plots
    }
    ## Diagnostic plots of results (imputed vs truth)
    # for(c in miss_cols){
    #   png(sprintf("%s/%s_col%d_NIMIWAE_%s%s_rz%s.png",diag_dir_name,mechanism,c,yesbeta,arch,yesrz))
    #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
    #                 x2=res_NIMIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
    #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
    #                 title=sprintf("NIMIWAE Column%d: True vs Imputed missing and observed values",c))
    #   dev.off()
    # }

    rm("res_NIMIWAE")  # if running multiple methods, save memory by saving results and removing from environment
  }

  #### MIWAE  (baseline comparison) ####
  if(grepl("^MIWAE",run_method)){
    print("MIWAE")
    rdeponz=FALSE;  covars_r=NULL; dec_distrib="Normal"; learn_r=NULL
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MIWAE_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MIWAE = NIMIWAE::tuneHyperparams(FUN=run_MIWAE,dataset=dataset,method="MIWAE",data=data,Missing=Missing,g=g,
                                  rdeponz=rdeponz, learn_r=learn_r,
                                  phi0=phi0,phi=phi,
                                  covars_r=covars_r)
      res_MIWAE$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MIWAE,file=sprintf("%s",fname0))
      #rm("res_MIWAE")
    }else{
      # load(fname0)
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_MIWAE.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
      #                 x2=res_MIWAE$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("MIWAE Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_MIWAE")
  }

  if(grepl("^HIVAE",run_method) | grepl("^VAEAC",run_method)){
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
    MissingDatas$test = MissingDatas$train
  }

  ### HIVAE ###
  if(grepl("^HIVAE",run_method)){
    print("HIVAE")
    dir_name2=sprintf("%s/%s/miss%d",dir_name,mechanism,miss_pct)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2,recursive = T),F)
    fname0=sprintf("%s/res_%s_%s_%d.RData",dir_name,run_method,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_HIVAE = NIMIWAE::tuneHyperparams(FUN=run_HIVAE,dataset=dataset,method="HIVAE",data=data,Missing=Missing,g=g,
                                  data_types=data_types,dir_name=dir_name2)
      res_HIVAE$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_HIVAE,file=sprintf("%s",fname0))
      #rm("res_HIVAE")
    }else{
      load(fname0)
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_HIVAE.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
      #                 x2=res_HIVAE$data_reconstructed[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("HIVAE Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_HIVAE")
  }

  ### VAEAC ###
  if(grepl("^VAEAC",run_method)){
    print("VAEAC")
    dir_name2=sprintf("%s/%s/miss%d",dir_name,mechanism,miss_pct)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2,recursive = T),F)
    fname0=sprintf("%s/res_%s_%s_%d.RData",dir_name,run_method,mechanism,miss_pct)
    print(fname0)
    # n_hidden_layers = if(dataset%in% c("TOYZ","TOYZ2","BANKNOTE","IRIS","WINE","BREAST","YEAST","CONCRETE","SPAM","ADULT","GAS","POWER")){10L} # default
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_VAEAC = NIMIWAE::tuneHyperparams(FUN=run_VAEAC,dataset=dataset,method="VAEAC",data=data,Missing=Missing,g=g,
                                  one_hot_max_sizes=one_hot_max_sizes, MissingDatas=MissingDatas, dir_name=dir_name2)
      res_VAEAC$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_VAEAC,file=sprintf("%s",fname0))
      #rm("res_VAEAC")
    }else{
      # load(fname0)
      # xhat_all = res_VAEAC$result    # this method reverses normalization intrinsically
      # # average imputations
      # xhat = matrix(nrow=nrow(datas$test),ncol=ncol(datas$test))
      # n_imputations = res_VAEAC$train_params$n_imputations
      # for(i in 1:nrow(datas$test)){ xhat[i,]=colMeans(xhat_all[((i-1)*n_imputations+1):(i*n_imputations),]) }
      #
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_VAEAC.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
      #                 x2=xhat[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("VAEAC Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_VAEAC")
  }

  ### MF ###
  if(grepl("^MF",run_method)){
    print("MF")
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MF_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MF = NIMIWAE::tuneHyperparams(FUN=run_missForest,dataset=dataset,method="MF",data=data,Missing=Missing,g=g)
      res_MF$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MF,file=sprintf("%s",fname0))
      #rm("res_MF")
    }else{
      # load(fname0)
      # if(is.null(res_MF$xhat_rev)){res_MF$xhat_rev = reverse_norm_MIWAE(res_MF$xhat_mf,norm_means,norm_sds)}
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_MF.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing",
      #                 x2=res_MF$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("MF Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_MF")
  }

  ### MEAN ###
  if(grepl("^MEAN",run_method)){
    print("MEAN")
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MEAN_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      t0=Sys.time()
      res_MEAN = NIMIWAE::tuneHyperparams(FUN=run_meanImputation,dataset=dataset,method="MEAN",data=data,Missing=Missing,g=g)
      res_MEAN$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MEAN,file=sprintf("%s",fname0))
      #rm("res_MEAN")
    }else{
      # load(fname0)
      # if(is.null(res_MEAN$xhat_rev)){res_MEAN$xhat_rev = reverse_norm_MIWAE(res_MEAN$xhat_mean,norm_means,norm_sds)}
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_MEAN.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
      #                 x2=res_MEAN$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("MEAN Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_MEAN")
  }

  ### MICE ###
  if(grepl("^MICE",run_method)){
    print(run_method)
    run_MICE=function(){}  # dummy function. Hard-coded MICE inside tuneHyperparams
    dir_name2=sprintf("%s",dir_name)
    ifelse(!dir.exists(dir_name2),dir.create(dir_name2),FALSE)
    fname0=sprintf("%s/res_MICE_%s_%d.RData",dir_name2,mechanism,miss_pct)
    print(fname0)
    if(!file.exists(fname0)){
      data0=datas$test; data0[Missings$test==0]=NA
      t0=Sys.time()
      res_MICE = mice::mice(data0, m=n_imputations)  # 25 default imputations
      # xhat = mice::complete(res_MICE)
      list_xhats = list()
      for(ii in 1:res_MICE$m){   # default at m=5 imputations by mice
        list_xhats[[ii]] = mice::complete(res_MICE,ii)
      }
      xhat = Reduce("+", list_xhats)/length(list_xhats)

      res_MICE$xhat = xhat
      res_MICE$time=as.numeric(Sys.time()-t0,units="secs")
      save(res_MICE,file=sprintf("%s",fname0))
      #rm("res_MEAN")
    }else{
      # load(fname0)
      # if(is.null(res_MEAN$xhat_rev)){res_MEAN$xhat_rev = reverse_norm_MIWAE(res_MEAN$xhat_mean,norm_means,norm_sds)}
      # for(c in miss_cols){
      #   png(sprintf("%s/%s_col%d_MEAN.png",diag_dir_name,mechanism,c))
      #   overlap_hists(x1=datas$test[Missings$test[,c]==0,c],lab1="Truth (missing)",
      #                 x2=res_MEAN$xhat_rev[Missings$test[,c]==0,c],lab2="Imputed (missing)",
      #                 x3=datas$test[Missings$test[,c]==1,c],lab3="Truth (observed)",
      #                 title=sprintf("MEAN Column%d: True vs Imputed missing and observed values",c))
      #   dev.off()
      # }
    }
    rm("res_MICE")
  }
}

ifelse(!dir.exists("./Results"), dir.create("./Results",recursive=T), F)
mechanisms=c("MCAR","MAR","MNAR")
run_methods=c("NIMIWAE_sup","NIMIWAE_unsup","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE_sup","MICE_unsup")

# Proof of Concept (fixed miss_pct)
# for(a in 1:length(mechanisms)){
#   runComparisons(dataset="SIM", sim_params=list(N=1e5, D=1, P=2, seed=NULL), save.dir="./Results", save.folder="SIM1", mechanism=mechanisms[a])
# }

# Supplementary simulations (P=8, 5 sims, vary miss_pct), proof of concept
# sim_indexes=1:5; miss_pcts = c(15,25,35)
sim_indexes=1; miss_pcts=25
for(a in 1:length(mechanisms)){for(b in 1:length(miss_pcts)){for(c in 1:length(sim_indexes)){for(d in 1:length(run_methods)){
  runComparisons(dataset="SIM", sim_params=list(N=1e5, D=2, P=8, seed=NULL), save.dir="./Results",save.folder="SIM1",
                 mechanism=mechanisms[a],miss_pct=miss_pcts[b], sim_index=sim_indexes[c],
                 rdeponz = c(F), arch = c("IWAE"), ignorable=F, run_method=run_methods[d])
}}}}

### Add varying N and varying phi

# UCI datasets (fixed miss_pct)
datasets=c("BANKNOTE","CONCRETE","RED","WHITE")
for(a in 1:length(mechanisms)){for(d in 1:length(datasets)){for(e in 1:length(run_methods)){
  runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a],rdeponz = c(F), arch = c("IWAE"),
                 ignorable=F, run_method=run_methods[e])
  if(grepl("NIMIWAE",run_methods[e])){
    runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a],rdeponz = c(F), arch = c("IWAE"),run_method=run_methods[e],
                 ignorable=T)
  }
}}}

run_methods=c("NIMIWAE_sup","NIMIWAE_unsup","MIWAE","HIVAE","VAEAC","MEAN","MICE_sup","MICE_unsup")

# Main simulations (P=100, 5 sims)
sim_indexes=1:5; miss_pcts = 25; phi_0s = c(1,5,10)
for(a in 1:length(mechanisms)){for(b in 1:length(miss_pcts)){for(c in 1:length(sim_indexes)){for(d in 1:length(run_methods)){for(e in 1:length(phi_0s)){
  runComparisons(dataset="SIM", sim_params=list(N=1e5, D=25, P=100, seed=NULL), save.dir="./Results",save.folder="SIM2",
                 mechanism=mechanisms[a],miss_pct=miss_pcts[b], sim_index=sim_indexes[c], phi_0=phi_0s[e],
                 rdeponz = c(F), arch = c("IWAE"), ignorable=F, run_method=run_methods[d])
}}}}}

# Nonlinear simulations
sim_indexes=1:5; miss_pcts = 25
for(a in 1:length(mechanisms)){for(b in 1:length(miss_pcts)){for(c in 1:length(sim_indexes)){for(d in 1:length(run_methods)){
  runComparisons(dataset="SIM_Nonlinear", sim_params=list(N=1e5, D=25, P=100, seed=NULL), save.dir="./Results",save.folder="SIMNL",
                 mechanism=mechanisms[a],miss_pct=miss_pcts[b], sim_index=sim_indexes[c],
                 rdeponz = c(F), arch = c("IWAE"), ignorable=F, run_method=run_methods[d])
}}}}

datasets=c("HEPMASS","POWER") # large datasets: runs into memory issues with missForest (MF)
for(a in 1:length(mechanisms)){for(d in 1:length(datasets)){for(e in 1:length(run_methods)){
  runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a], run_method=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MICE"),
                 rdeponz = c(F), arch = c("IWAE"), ignorable=F, run_method=run_methods[e])
  if(grepl("NIMIWAE",run_methods[e])){
    runComparisons(dataset=datasets[d], save.dir="./Results",mechanism=mechanisms[a], run_method=c("NIMIWAE"),
                 rdeponz = c(F), arch = c("IWAE"), ignorable=T, run_method=run_methods[e])
  }
}}}
#### NEED TO INCLUDE IGNORABLE RUN IN PARENTHESES ####

# Physionet analysis
run_methods=c("NIMIWAE_sup","NIMIWAE_unsup","MIWAE","HIVAE","VAEAC","MEAN","MF","MICE_sup","MICE_unsup")
for(a in 1:length(run_methods)){
  runComparisons(dataset="Physionet", save.dir="./Results",mechanism="MNAR", ignorable=F, run_method=run_methods[a])  # for Physionet, missingness isn't simulated. Inherent missingness is assumed to be MNAR
  if(grepl("NIMIWAE",run_methods[a])){
    runComparisons(dataset="Physionet", save.dir="./Results",miss_pct=NA, mechanism="MNAR", run_method=run_methods[a], ignorable=T)  # for Physionet, missingness isn't simulated. Inherent missingness is assumed to be MNAR
  }
}
