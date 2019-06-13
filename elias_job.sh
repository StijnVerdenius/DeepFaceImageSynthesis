#!/bin/bash
#Set job requirements
#SBATCH --job-name=face_synth_with_landmarks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

#Loading modules
module load Miniconda3/4.3.27
module load CUDA/9.0.176
module load cuDNN/7.3.1-CUDA-9.0.176

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib6$
export PYTHONIOENCODING=utf8

echo "copy directory"
mkdir $TMPDIR/lgpu0293
cp -r $HOME/ProjectAI/DeepFakes $TMPDIR/lgpu0293

echo "cd inwards"
cd $TMPDIR/lgpu0293/DeepFakes

echo "activate env"
source activate dl

echo " ------ Job is started ------- "
echo " "

srun python3 main.py

cp -r $TMPDIR/lgpu0293/DeepFakes/results/output/* $HOME/ProjectAI/DeepFakes/results/output

echo " "
echo " ------ Job is finished -------"
