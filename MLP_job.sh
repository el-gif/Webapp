#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=16:mem=16gb
#PBS -q v1_medium24
#PBS -N MLP_Training
#PBS -o job_output.log
#PBS -e job_error.log

# Arbeitsverzeichnis wechseln
cd $PBS_O_WORKDIR

# Miniforge-Umgebung aktivieren
export PATH="/rds/general/user/abp224/home/miniforge3/bin:$PATH"
source /rds/general/user/abp224/home/miniforge3/etc/profile.d/conda.sh
conda activate webapp_env_conda

# Eingabedateien kopieren
cp $HOME/MLP_KFold.ipynb $TMPDIR
cp $HOME/WPPs+production+wind.json $TMPDIR
cd $TMPDIR

cat /proc/cpuinfo

# Notebook ausführen
/usr/bin/time -v jupyter nbconvert --to notebook --execute MLP_KFold.ipynb --output MLP_KFold_Executed.ipynb

# Ergebnisse kopieren
cp MLP_KFold_Executed.ipynb $HOME

notify-send "Job completed!" "Your HPC job has finished."
echo "Job finished!"