source("processComparisons.R")

# Proof of Concept
saveFigures(datasets="SIM1", sim_index=1, miss_pcts=25)
saveFigures(datasets="SIM2")
saveFigures(datasets=c("BANKNOTE","CONCRETE","RED","WHITE", sim_index=1))
saveFigures(datasets=c("HEPMASS","POWER"), sim_index=1, methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN"))   # huge datasets can't run MF b/c memory

df = tabulate_UCI(mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(25),datasets=c("BANKNOTE","CONCRETE","HEPMASS","POWER","RED","WHITE"),
                  methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi0=5,sim_index=1)
save(df,file="Results/UCI_df.RData")   # vanilla no penalty run
