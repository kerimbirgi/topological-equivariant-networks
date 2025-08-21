#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=24:0:0
#PBS -N rcut4

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate etnn

python create_bindingnet.py dataset=subset_20p_no_supercell_crossconnect_rcut_3
