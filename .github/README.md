# ML-Tech-Cheatsheet 📄

Personal "cheatsheet" repository for my ideal machine learning tech-stack. I use this repository to play around and
familiarize
with ML libraries, advanced git and GitHub features, virtualization and so on 🤓.

## Table of Contents 📜

1. [IDEs Plugins 🧰](#IDEs-plugins-🧰)
    1. [VSCode](#VSCode)
    2. [PyCharm](#PyCharm)
2. [Machine Learning Libraries 🤖](#Machine-Learning-Libraries-🤖)
    1. [The classics](#The-classics)
    2. [Pytorch, Lightning and W&Bs](#Pytorch,-Lightning-and-W&Bs)
    3. [DeepSpeed](#DeepSpeed)
    4. [albumentations](#albumentations)
    5. [einops](#einops)
    6. [gradio and streamlit](#gradio-and-streamlit)
3. [Environments 🌎](#Environments-🌎)
    1. [conda](#conda)
    2. [docker](#docker)
4. [CLI Utilities 👨‍💻](#CLI-Utilities-👨‍💻)
5. [High Performance Computing 🦾](#High-Performance-Computing-🦾)
    1. [slurm](#slurm)
6. [Git 🐱](#Git-🐱)
    1. [Protected Branches](#Protected-Branches)
    2. [Tags and Releases](#Tags-and-Releases)
    3. [LFS](#LFS)
    4. [Hidden Directory](#Hidden-Directory)
    5. [GitHub Actions](#GitHub-Actions)
    6. [GitHub Pages](#GitHub-Pages)

## IDEs plugins 🧰

### VSCode

* Python
* RainbowCSV
* Remote
* CoPilot
* GitLens
* Docker
* Jupiter
* Gitignore
* vscode-pdf

### PyCharm

* GitToolBox
* CoPilot
* Docker

## Machine Learning Libraries 🤖

### The classics

* **NumPy** - Math operations, manipulations, linear algebra and more.
* **Pandas** - Tabular data management.
* **MatplotLib** and **Seaborn** - All sorts of plots.
* **OpenCV2**, **Pillow**, and **Sci-Kit Image** - Image manipulation

### Pytorch, Lightning and W&Bs

[PyTorch](https://pytorch.org/) is currently the reference ML framework for Python.

[Weights and Biases](https://wandb.ai/) (W&B) allows to easily track experiments, performances, parameters and so on in
a single place.

[PyTorch Lightning](https://www.pytorchlightning.ai/) gets rid of most of the usual PyTorch boilerplate code, like
train/val/test loops, backward and optim steps and so on. It also allows to easily use powerful pytorch features and
other libraries (like W&B) by inserting just few optional parameters here and there.

### DeepSpeed
[DeepSpeed](https://www.deepspeed.ai/) allows for distributed high-performance and efficient training. 
DeepSpeed is [supported in PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed).

### albumentations

All sorts of popular [image augmentations](https://github.com/albumentations-team/albumentations#list-of-augmentations),
like ColorJitter, ZoomBlur, Gaussian Noise... are implemented by [albumentations](https://albumentations.ai/).

### einops

Manipulation of tensors (reshaping, concatenating, ...) with [einops](https://github.com/arogozhnikov/einops) is
extremely intuitive and time-saving.

### gradio and streamlit

To quickly create interactive apps based on trained machine learning models, [gradio](https://gradio.app/)
and [streamlit](https://streamlit.io/) are among the most popular frameworks.

## Environments 🌎

### conda

[Conda](https://conda.io/) allows to easily create and share virtual environments. The
command `conda env export > environment.yml` creates a .yml file that can be used to create an identical virtual
environment.

### Docker

[Docker](https://docker.com) allows to emulate a whole operating system.

## CLI Utilities 👨‍💻

* `nvidia-smi` ➡️ Check NVIDIA Cards current status
* `ps`, `top`, `htop` ➡️Check currently running processes
* [`nvitop`](https://github.com/XuehaiPan/nvitop) ➡️Like `nvidia-smi`, but better.
* [`tmux`](https://github.com/tmux/tmux/wiki) ➡️Terminal multiplexer, allows to easily detach jobs.
* `~/.ssh/config` and `~/.ssh/authorized_keys` files to define known host names and authorized ssh keys.

## High Performance Computing 🦾

### slurm

HPC clusters typically use a cluster management and job scheduling tool. [Slurm](https://slurm.schedmd.com/) allows to
schedule jobs, handle priorities, design partitions and much more. Cheatsheet files for slurm are under
the [/slurm ](/slurm) folder.

## Git 🐱

Taking the time to go through most of [GitHub's Documentation](https://docs.github.com/) at least once is very
important. Here's a few features to keep in mind.

### Protected Branches

[Protected branches](https://docs.github.com/en/rest/branches/branch-protection)

### Tags and Releases

Important commits can be tagged. Then, jumping to a commit is easy as

### LFS

[Git Large File System](https://git-lfs.github.com/) allows to push bigger files to the GitHub repository. **Careful**:
There is a global usage quota per GitHub account that goes across repositories.

### Hidden Directory

The `.github` directory allows to keep the landing page of the GitHub repository "clean" and includes:

* **[CONTRIBUTING.md](CONTRIBUTING.md)** ➡️ Guidelines to contribute to the repository.
* **[ISSUE_TEMPLATE.md](ISSUE_TEMPLATE.md)** ➡️ Template for issues.
* **[PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md)** ➡️Template for pull requests.
* **[README.md](README.md)** ➡️Repository's README (i.e. this) file.
* **[workflows](workflows)** ➡️Directory which contains .yaml files for GitHub actions.

### GitHub Actions

[GitHub Actions](https://docs.github.com/en/actions) allows to execute custom actions automatically upon some triggers
by some events (pull requests, pushes, issues opened, ...).

### GitHub Pages

[GitHub Pages](https://docs.github.com/en/pages) allows to host a webpage for each GitHub repository.
