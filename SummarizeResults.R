source("processComparisons.R")

# Proof of Concept
saveFigures(datasets="SIM1", sim_index=1, miss_pcts=25, outfile="Results/SFig1.png")

# Suppl Simulations (P=8)
saveFigures(datasets="SIM2", outfile="Results/SFig2.png")

# Main Simulations (P=100)
saveFigures(datasets="SIM3", outfile="Results/Fig2.png")

# UCI (figures for diagnostic purposes. Not in paper. Outfile goes to default directory Results/<dataset>/phi5/sim<sim_index>/)
saveFigures(datasets=c("BANKNOTE","CONCRETE","RED","WHITE", sim_index=1))
saveFigures(datasets=c("HEPMASS","POWER"), sim_index=1, methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN"))   # huge datasets can't run MF b/c memory

df = tabulate_UCI(mechanisms=c("MCAR","MAR","MNAR"),miss_pct=c(25),datasets=c("BANKNOTE","CONCRETE","HEPMASS","POWER","RED","WHITE"),
                  methods=c("NIMIWAE","MIWAE","HIVAE","VAEAC","MEAN","MF"),phi0=5,sim_index=1)
save(df,file="Results/Tab1.RData")   # vanilla no penalty run

## UCI


# Physionet analysis
## Fig3abc (boxplots, table, scatterplots)
