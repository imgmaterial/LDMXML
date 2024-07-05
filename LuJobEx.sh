#!/bin/bash
#
#SBATCH -A lu2024-2-28 #This line tells the system that an job is to be submitted, 'lu2022-2-58' determines what project you are part of and this is the LDMX project 
#SBATCH -t 168:00:00 -p lu48  #This system tells the system how much time the job should have on Aurora, so -t 18:00:00 indicates that the job has 18 hours to be completed but if finished before that the job will be considered complete 
#SBATCH --mem-per-cpu=10000 #This tells the system how much memory should be used per CPU involved in the calculation of the job 
#SBATCH -N 1 #This specifies how many nodes should be used for the job 
python3 EcalCNNPooled.py #This is how you actually tell the system which python file to run so simply implement whichever file is to be run 
