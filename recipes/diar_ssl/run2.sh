#module load miniforge3


# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
#cd /home/users/ntu/bhargavi/Diarizen/recipes/diar_ssl
#set -eu
#ulimit -n 2048

# general setup
stage=1
recipe_root=/home/users/ntu/bhargavi/Diarizen/recipes/diar_ssl
exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

# training setup
use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
train_conf=$conf_dir/wavlm_updated_conformer.toml
# train_conf=$conf_dir/wavlm_frozen_conformer.toml
# train_conf=$conf_dir/fbank_conformer.toml
# train_conf=$conf_dir/pyannote_baseline.toml

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test
data_dir=$recipe_root/data/AMI_AliMeeting_AISHELL4
seg_duration=8





# =======================================
# =======================================
if [ $stage -le 1 ]; then
    if (! $use_dual_opt); then
        echo "stage1: use single-opt for model training..."
        conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
            --num_processes 2 --main_process_port 1134 \
            run_single_opt.py -C $train_conf -M validate
    else
        echo "stage1: use dual-opt for model training..."
        accelerate launch \
            --num_processes 1 --main_process_port 1134 \
            run_dual_opt.py -C $train_conf -M train
    fi
fi


