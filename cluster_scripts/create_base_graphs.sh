#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:0:0
#PBS -N create_base_graphs

cd $PBS_O_WORKDIR
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate etnn

python preprocess/single_graph_processing.py

