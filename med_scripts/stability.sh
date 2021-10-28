#!/bin/bash

cpu_def=0
cpu_def1=1
set_id=0
criteria="maxsum"


trap "exit" INT
for prop in $(find ./gen_train_datasets/set$set_id/ -name "idx*"); 
do
    target=$prop".txt"
    target=${target/datasets/results}
    target=${target/set$set_id\//}
    target=${target/idx/$criteria"_idx"}
    echo $target
    if [ -f "$target" ]; then
            echo "$target file is done"
    else
            #echo "python"

            taskset --cpu-list $cpu_def,$cpu_def1 python ./med_experiments/stability_analysis.py --prop $prop --record --criteria $criteria 
    fi
done


