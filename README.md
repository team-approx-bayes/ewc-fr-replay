# Improving Continual Learning by Accurate Gradient Reconstructions of the Past

This repository contains the code for the paper [Improving Continual Learning by Accurate Gradient Reconstructions of the Past](https://openreview.net/forum?id=b1fpfCjja1), TMLR 2023.

![EWC+FR+Replay](fig1.png?raw=true)

## Setup
Clone this repo and install the dependencies in `requirements.txt` in a Python 3.9 environment.
For convenience, add the repo's `src` folder to your Python path.

For running ImageNet experiments, install `ffcv` following [their instructions](https://github.com/libffcv/ffcv).
Also, run the command below to convert the ImageNet dataset to the FFCV format -- see the [ffcv-imagenet repo](https://github.com/libffcv/ffcv-imagenet) for details (this requires cloning `ffcv-imagenet` and setting the environment variables in `convert_imagenet_ffcv.sh`).
```console
$ python src/data/convert_imagenet_ffcv.sh
```

## Run Experiments
Run the command below to create the experiment config files corresponding to the methods reported in the paper (placed in `src/configs`).
```console
$ python src/create_exp_config_files.py
```

Run the command below to launch an experiment with a given config (see `src/configs`).
```console
$ python src/main.py --config path/to/config.yaml
```

## Plot Results
Run the command below to generate the plots shown in the paper (placed in `src/plotting/plots`).
```console
$ python src/plotting/plot_results.py
```

## Citation
```
@article{daxberger2023improving,
    title={Improving Continual Learning by Accurate Gradient Reconstructions of the Past},
    author={Erik Daxberger and Siddharth Swaroop and Kazuki Osawa and Rio Yokota and Richard E Turner and Jos{\'e} Miguel Hern{\'a}ndez-Lobato and Mohammad Emtiyaz Khan},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=b1fpfCjja1},
}
```

## Acknowledgements
- [CLSurvey](https://github.com/Mattdl/CLsurvey) for TinyImageNet related code.
- [ffcv](https://github.com/libffcv/ffcv) and [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet) for ImageNet related code.