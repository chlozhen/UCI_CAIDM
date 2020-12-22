#!/bin/bash
 
# --- Create (parameter.csv) and (logs/output directory)
python params.py

# --- CD into the folder that has the model.py file and hyper.csv, create scripts directory
jarvis script -jmodels "./" -py "model" -csv "hyparams/TR3_alpha" -output_dir "./scripts/scripts-TR3_alpha" -name "exp" 

# --- CD into scripts folder and run this to add all the scripts into jarvis cluster - run models! save weights!
jarvis cluster add -scripts "./scripts/scripts-TR3_alpha/*.sh" -workers "[gr]tx"

# --- Kill clusters
#jarvis cluster kill -workers [name_of_worker]