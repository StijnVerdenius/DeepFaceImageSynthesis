#!/bin/bash
#Set job requirements
#SBATCH --job-name=face_synth_with_landmarks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

#Loading modules
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib6$
export PYTHONIOENCODING=utf8

echo "copy directory"
mkdir $TMPDIR/lgpu0386
cp -r $HOME/DeepFakes $TMPDIR/lgpu0386

echo "cd inwards"
cd $TMPDIR/lgpu0386/DeepFakes

echo "activate env"
source activate dl

echo " ------ Job is started ------- "
echo " "

srun python3 main.py

cp -r $TMPDIR/lgpu0386/DeepFakes/results $HOME/DeepFakes/results

echo " "
echo " ------ Job is finished -------"