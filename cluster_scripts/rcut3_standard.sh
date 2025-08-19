#PBS -l select=1:ncpus=4:mem=80gb:ngpus=1
#PBS -l walltime=24:0:0
#PBS -N rcut3_standard

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate etnn

python main_bindingnet.py dataset=subset_20p_no_supercell_crossconnect_rcut_3 experiment=standard

