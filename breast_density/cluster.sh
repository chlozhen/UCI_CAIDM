#!/bin/bash
 
# --- Create (parameter.csv) and (logs/output directory)
python params.py

# --- CD into the folder that has the model.py file and hyper.csv, create scripts directory
jarvis script -jmodels "./" -py "model" -csv "hyparams/hyper_lrt_sched" -output_dir "./scripts/scripts-hyper_lrt_sched" -name "exp" 

# --- CD into scripts folder and run this to add all the scripts into jarvis cluster - run models! save weights!
jarvis cluster add -scripts "./scripts/scripts-hyper_lrt_sched/*.sh" -workers "[gr]tx"

# --- Kill clusters
#jarvis cluster kill -workers [name_of_worker]