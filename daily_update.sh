#!/bin/bash
source /home/anlabadmin/miniconda3/etc/profile.d/conda.sh && conda activate cuda
cd /home/anlabadmin/Documents/Lashinbang-test/create-index
python update_index.py