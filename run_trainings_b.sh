#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=23:00:00
#SBATCH --job-name=b_job
#SBATCH --output=b_job_%j.txt
#SBATCH --mail-type=ALL

cd $SCRATCH/AP-bIO 
source berjlab_ml/bin/activate
module load python/3.11.5

python main.py -s -y params_1.yml
python main.py -s -y params_1.yml
python main.py -s -y params_1.yml
python main.py -s -y params_1.yml
python main.py -s -y params_1.yml