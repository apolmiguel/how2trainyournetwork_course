# How to Train Your Network Course

## Introduction
Welcome to the How to Train Your Network course. If you decide to use `PANNA` in the terminal for the exercises, this README will guide you through setting up your environment and installing the necessary packages. 

This README will guide you through setting up your environment and installing the necessary packages.

## Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

## Setting Up a Virtual Environment
We recommend to create and use a virtual environment for using this specific version of PANNA, as we will have to revert to an older version of some packages (i.e.`tensorflow` needs to be at version `2.13.0` for the tutorial to work). Follow these steps to set it up: 


It is recommended to use a virtual environment to manage your project dependencies. Follow the steps below to set up a virtual environment:

1. **Create a python virtual environment (venv)**:

```sh
python -m venv panna_venv
```

2. **Activate the virtual environment**:

    - On macOS and Linux:
        ```sh
        source panna_venv/bin/activate
        ```
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```


## Installing the `panna` Package
Once the virtual environment is activated, you can install the `panna` package using pip:

```sh
pip install panna
```

## Deactivating the Virtual Environment
After you have finished working, you can deactivate the virtual environment by running:

```sh
deactivate
```

## Additional Resources
- [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [panna Documentation](https://panna.readthedocs.io/)

## Contact
For any questions or issues, please contact the course instructor at [instructor@example.com](mailto:instructor@example.com).