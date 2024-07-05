#!/bin/bash
#
#SBATCH -A lu2024-2-28 #This line tells the system that an job is to be submitted, 'lu2022-2-58' determines what project you are part of and this is the LDMX project 
#SBATCH -t 100:00:00 -p lu48  #This system tells the system how much time the job should have on Aurora, so -t 18:00:00 indicates that the job has 18 hours to be completed but if finished before that the job will be considered complete 
#SBATCH --mem-per-cpu=6000 #This tells the system how much memory should be used per CPU involved in the calculation of the job 
#SBATCH -N 1 #This specifies how many nodes should be used for the job 
echo Test Sample Processing
apptainer exec -B /projects/hep/fs9/shared/ldmx/users/pa8701os:/data ../ldmx-container/ldmx-sw_v3.3.3.sif python3 CNNEcalPreTrigPar.py
#python3 CNNEcalPre1e.py #This is how you actually tell the system which python file to run so simply implement whichever file is to be run 
echo done