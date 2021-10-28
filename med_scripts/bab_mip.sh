#!/bin/bash

cpus_total=10
cpu_def=0
cpu_rel=0
timeout=3600
pdprops='m2_easy.pkl'
flags='--record --gurobi --bab_kw'

taskset --cpu-list $cpu_def python med_experiments/med_bab_mip.py --cpu_id $cpu_rel --timeout $timeout --cpus_total $cpus_total --pdprops $pdprops $flags
