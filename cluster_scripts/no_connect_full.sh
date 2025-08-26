#PBS -l select=1:ncpus=4:mem=80gb:ngpus=1
#PBS -l walltime=24:0:0
#PBS -N no_connect

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate etnn

python main_bindingnet.py dataset=full_no_supercell_no_crossconnect experiment=standard

