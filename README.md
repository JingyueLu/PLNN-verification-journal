# Branch and Bound for Piecewise Linear Neural Network Verification
This repository contains all the code necessary to replicate the experiments
reported in the [journal paper](https://arxiv.org/pdf/1909.06588.pdf). 

## Dependences
* This code is developed on the original implementations of Branch and Bound methods, provided in the github package [PLNN_verification](https://github.com/oval-group/PLNN-verification). We have also directly used the MIPplanet solver provided in  [PLNN_verification](https://github.com/oval-group/PLNN-verification).
  
## Installation
We recommend installing everything into a virtual environment. The detailed installation instruction can be found [here](https://github.com/oval-group/PLNN-verification).

## Running the experiments
* experiments related files are included in the folder ./med_experiments/. In detail, the stability script is used for generating training datasets while the med_bab_mip contains the main function for running all experiments. Experiment settings are introduced through flags.
* bash scripts for experiments are also provided and they are in the folder ./med_scripts/.


## Reference
If you use this work in your research, please cite:

```
@Article{bunel2020branch,
      title={Branch and Bound for Piecewise Linear Neural Network Verification}, 
      author={Rudy Bunel and Jingyue Lu and Ilker Turkaslan and Philip H. S. Torr and Pushmeet Kohli and M. Pawan Kumar},
      year={2020},
      journal={Journal of Machine Learning Research}
}
```
