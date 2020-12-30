source("processComparisons.R")

# Proof of Concept
saveFigures(datasets="SIM1", sim_index=1, miss_pcts=25, outfile="Results/SFig1.png",
            methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"))

# Suppl Simulations (P=8)
saveFigures(datasets="SIM2", outfile="Results/SFig2.png",
            methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"))

# Main Simulations (P=100)
saveFigures(datasets="SIM3", outfile="Results/Fig2.png",
            methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"))

# UCI (figures for diagnostic purposes. Not in paper. Outfile goes to default directory Results/<dataset>/phi5/sim<sim_index>/)
saveFigures(datasets=c("BANKNOTE","CONCRETE","RED","WHITE"), sim_index=1, methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"))
saveFigures(datasets=c("HEPMASS","POWER"), sim_index=1, methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN"))   # huge datasets can't run MF b/c memory

df = tabulate_UCI(mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(25),datasets=c("BANKNOTE","CONCRETE","HEPMASS","POWER","RED","WHITE"),
                  methods=c("IMIWAE","NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi0=5,sim_index=1)
save(df,file="Results/Tab1.RData")   # vanilla no penalty run

## UCI


# Physionet analysis

## Table 1: descriptive stats (missingness proportion of each feature):
## runComparisons.R must have been run first for Physionet --> raw data must have been organized and processed already
descriptive_stats=function(){
  library(reticulate)
  np <- import("numpy")


  features = c('ALP','ALT','AST','Albumin','BUN','Bilirubin',
               'Cholesterol','Creatinine','DiasABP','FiO2','GCS',
               'Glucose','HCO3','HCT','HR','K','Lactate','MAP', #'MechVent',
               'Mg','NIDiasABP','NIMAP','NISysABP','Na','PaCO2',
               'PaO2','Platelets','RespRate','SaO2','SysABP','Temp',
               'TroponinI','TroponinT','Urine','WBC','pH')

  npz1 <- np$load("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/data_train_val.npz")
  npz_test <- np$load("data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/data_test.npz")
  d_train = npz1$f$x_train; d_val = npz1$f$x_val; d_test = npz_test$f$x_test
  M_train = npz1$f$m_train; M_val = npz1$f$m_val; M_test = npz_test$f$m_test

  library(abind)
  d3=abind(d_train, d_val, d_test, along = 1)
  M3=abind(M_train, M_val, M_test, along = 1)

  d3=aperm(d3, c(2,1,3)); M3=aperm(M3,c(2,1,3))
  d = matrix(d3, nrow=dim(d3)[1]*dim(d3)[2], ncol=dim(d3)[3])
  M = 1-matrix(M3, nrow=dim(M3)[1]*dim(M3)[2], ncol=dim(M3)[3])

  d=d[,-19]; M=M[,-19]   # remove "MechVent"
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

  save(tab1,file="Physionet_descriptive_stats.out")
  return(tab1)
}
tab=descriptive_stats()
tab

## Figure 3abc (boxplots, table, scatterplots)
summarize_Physionet(prefix="",mechanism="MNAR",miss_pct=NA,dataset="Physionet")
