# Benchmarking Domain Adaptation for Chemical Processes on the Tennessee Eastman Process

## Introduction

In this repository, you may find the code for reproducing our paper "Benchmarking Domain Adaptation for Chemical Processes on the Tennessee Eastman Process". Our code relies on the simulations by Reinartz et al. [1, 2], [[which you may find here]](https://data.dtu.dk/articles/dataset/Tennessee_Eastman_Reference_Data_for_Fault-Detection_and_Decision_Support_Systems/13385936).

Concerning the benchmark, you currently have two options,

1. You may build the benchmark yourself.
2. You may download our preprocessed data at this link (to be released).

Note that, choosing 1 may lead to potentially different, but statistically equivalent results due to sampling and shuffling of the data. Choosing 2 allows you to reproduce our results at a more accurate level. We chosen to provide the first choice for extensibility, and for transparency of the data pre-processing steps.

For building the benchmark, you can either use our downloader code (see ```/benchmark/preprocessing/downloader.py```), or, if you prefer, you can download the data from source and put it in ```/tep_data/raw```.

## About this benchmark

This benchmark is build upon the simulations of Reinartz et al., which are described in publications [1] and [2]. We strongly encourage users to read these papers, as well as look into their library, [PyTEP](https://github.com/ChristopherReinartz/pytep) for a deeper insight at how their simulations are done. The Tennessee Eastman process is a complex, large scale chemical process, first proposed by Downs and Vogel [3]. It consists of a set of exothermic reactions for the production of a set of products, under a controlled environment.

Previous work have considered fault detection and diagnosis with data from this chemical process [1]. Some [5] use domain adaptation for diagnosis, but they consider a smaller sub-set of faults, and do not provide a public, reproducible benchmark for the cross-domain fault diagnosis community.

## Building the benchmark

Building the benchmark from scratch means to use our code for downloading and pre-processing the data. You can do so in 2 steps,

- Import ```tep_data_downloader``` from ```benchmark/preprocessing```. You can call this function as follows,

```py
from benchmark.preprocessing import tep_data_downloader

tep_data_downloader(
    mode=1, destination_path='./tep_data/raw', chunk_size=1024)
```

which will download data from mode 1 on ```./tep_data/raw```. Note that downloading the raw data can take a long time, as the raw data from Reinartz et al. has over 142 GB. Alternatively, you can download the data yourself and place it in the appropriate folder. The next step consists of pre-processing the raw data, that is, building our benchmark.

- Import ```tep_build_benchmark``` from ```benchmark/preprocessing```. You can call this function as follows,

```py
from benchmark.preprocessing import tep_build_benchmark

tep_build_benchmark(
    in_path='./tep_data/raw',
    out_path='./tep_data/benchmark',
    variables=None
)
```

where ```in_path``` corresponds to the path where you stored the raw data from Reinartz et al, and ```out_path``` is the path where you want to write the benchmark. The argument ```variables``` defines which variables, out of the original 53 + 1 (the 1st variable is simply the simulation time) you want to keep. If you do not specify this argument, you build our benchmark, i.e., you consider XME(1) through XME(22) and XMV(1) through XMV(12) (total of 34 variables).

## Experiments

In this repository, you may find three kinds of experiments. These are located under ```./experiments/```, and take the form of jupyter notebooks.

- Exploratory Data Analysis,
- Source-only baselines
    - Single-source baseline
    - Multi-source baseline

We did not include experiments with domain adaptation, as these will be released in an upcoming library focused on domain adaptation.

### Reproducibility note

With our code, you may generate yourself the benchmark used in our paper. Based on the data, you can partition the data of each domain in 5 folds. For our experiments, we generated the indices based on a random shuffling of the generated data. By generating the 5-folds again, you may not find the same values, but these should be statistically comparable. We refer practicioners for the data used in our experiments [hosted in Kaggle](https://www.kaggle.com/datasets/eddardd/tennessee-eastman-process-domain-adaptation/data), for the exact reproduction of our results.

## Extending this benchmark

An advantage of open-sourcing the code for our benchmark is its extensibility. For instance, by changing the inner procesing logic of ```build_benchmark``` in ```benchmark/preprocessing``` you can test with different cross-domain fault diagnosis scenarios. Here are some questions that were not explored in our initial submission,

- Do algorithms generalize to different fault intensities?
- Do algorithms work well under incomplete simulations?
- Can the benchmark integrate setpoint variation and mode transition data?

Furthermore, note that our work can serve for cross-domain fault detection. Instead of using the whole historical data for classifying faults, methods could instead focus on cropping sub-segments of the signal and try modeling the no-fault scenario.

## Dependencies

```
h5py==3.7.0
numpy==1.23.5
pytorch_lightning==2.0.2
requests==2.32.3
torch==2.2.2
torchmetrics==0.10.3
tqdm==4.64.1
```

## References

[1] Reinartz, C., Kulahci, M., & Ravn, O. (2021). [An extended Tennessee Eastman simulation dataset for fault-detection and decision support systems](https://www.sciencedirect.com/science/article/pii/S0098135421000594). Computers & chemical engineering, 149, 107281.

[2] Reinartz, C., & Enevoldsen, T. T. (2022). [pyTEP: A Python package for interactive simulations of the Tennessee Eastman process](https://www.sciencedirect.com/science/article/pii/S2352711022000449). SoftwareX, 18, 101053.

[3] Downs, J. J., & Vogel, E. F. (1993). [A plant-wide industrial process control problem](https://www.sciencedirect.com/science/article/abs/pii/009813549380018I). Computers & chemical engineering, 17(3), 245-255.

[4] Bathelt, A., Ricker, N. L., & Jelali, M. (2015). [Revision of the Tennessee Eastman process model](https://www.sciencedirect.com/science/article/abs/pii/S2405896315010666). IFAC-PapersOnLine, 48(8), 309-314.

[5] Wu, H., & Zhao, J. (2020). [Fault detection and diagnosis based on transfer learning for multimode chemical processes](https://www.sciencedirect.com/science/article/pii/S0098135419308762). Computers & Chemical Engineering, 135, 106731.