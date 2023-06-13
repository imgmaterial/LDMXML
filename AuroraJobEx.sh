#!/bin/bash
#
#SBATCH -A 'lu2022-2-58' 
#SBATCH -t 18:00:00 -p lu
#SBATCH --mem-per-cpu=6000

python 1EpreTS.py
