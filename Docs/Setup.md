Environment Setup Guide

### 1. Create a Virtual Environment (Python 3.11) Ensure you have Python 3.11 installed. Run the following command in your project root to create the environment:
```Bash

# MacOS/Linux
python3.11 -m venv venv

# Windows
py -3.11 -m venv venv
```

### 2. Activate the Environment
```Bash

# MacOS/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```
### 3. Install PyTorch Since PyTorch installation varies heavily based on your operating system (Linux/Windows/Mac) and compute hardware (CUDA/ROCm/CPU), visit the official selector to get the correct installation command:

    URL: https://pytorch.org/get-started/locally/

Copy the generated pip command from that page and run it in your active environment before installing other dependencies.

### 4. Install CityLearn and Stable Baselines Once PyTorch is installed, run:
```Bash

pip install citylearn stable-baselines3
```