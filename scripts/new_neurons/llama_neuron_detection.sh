bash
source ~/.bashrc
cd /home/s2410121/proj_LA/activated_neuron/new_neurons
conda activate proj_LA_neuron_detection
module load cuda/12.1
python neuron_detection.py
conda deactivate
