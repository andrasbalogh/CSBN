# This is the "config" file. At the moment, only these variables can be
# adjusted but the user can choose to add or remove specific values.

gpuNum=5                        # GPU Device used

network = "ban_cpmtx"           # Comment out the desired network
# network = trnGamma_cpmtx 
# network = ern_cpmtx 
# network = trnExp_cpmtx

population_size = 100           # N

lt_trnExp = -40.0 # parameter for trn #-20 (does not work for all pop. sizes)
