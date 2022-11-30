#!/bin/sh


#SBATCH -p calc05
#SBATCH -J FCNwithQ_learning
#SBATCH --mem=32G
#SBATCH -o /home/kyousuke.senda/project/FCNA/Q_out.txt
#SBATCH -e /home/kyousuke.senda/project/FCNA/Q_error.txt

echo "FCNwithQ"

/home/kyousuke.senda/project/FCNA/run.sh kesu2
