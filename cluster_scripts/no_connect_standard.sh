#PBS -l select=1:ncpus=4:mem=80gb:ngpus=1
#PBS -l walltime=24:0:0
#PBS -N no_connect

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate etnn

python main_bindingnet.py dataset=subset_20p_no_supercell_no_connect_cross_self experiment=standard

