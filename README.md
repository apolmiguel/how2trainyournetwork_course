i# How to Train Your Network Course

## Introduction
Welcome to the How to Train Your Network course. If you decide to use `PANNA` in the terminal for the exercises, this README will guide you through setting up your environment and installing the necessary packages. The documentation of PANNA may be found at [https://pannadevs.gitlab.io/pannadoc/](https://pannadevs.gitlab.io/pannadoc/).

## Prerequisites
- Python `<3.12` preferably, as dependency problems may arise when loading tensorflow `v2.13.0`
- pip (Python package installer)
- (hopefully) a Linux or a macOS system.

## Cloning the repository
In your terminal, clone the repository using:

```sh
git clone https://github.com/apolmiguel/how2trainyournetwork_course.git
```

## Setting Up a Virtual Environment
We recommend to create and use a virtual environment for using this specific version of PANNA, as we will have to revert to an older version of some packages (i.e.`tensorflow` needs to be at version `2.13.0` for the tutorial to work). Follow these steps to set it up: 


It is recommended to use a virtual environment to manage your project dependencies. Follow the steps below to set up a virtual environment:

1. **Create a python virtual environment (venv)**:

```sh
python -m venv panna_venv
```

2. **Activate the virtual environment**:

On the root folder, `how2trainyournetwork_course/`, execute the following commands:

```sh
source panna_venv/bin/activate
```


## Installing the `panna` Package
Once the virtual environment is activated, go to `panna-master` and install `PANNA`:

```sh
cd panna-master/
pip install .
```


## Deactivating the Virtual Environment
After you have finished working, you can deactivate the virtual environment by running:

```sh
deactivate
```
