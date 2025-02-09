# Use Miniconda base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies manually
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    libgcc-s1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Mamba for faster dependency resolution
RUN conda install -n base -c conda-forge mamba

# Create Conda environment using Mamba
RUN mamba env create -f environment.yml

# Activate Conda environment
SHELL ["conda", "run", "-n", "depth-jitter", "/bin/bash", "-c"]

# Install any additional dependencies
RUN conda run -n depth-jitter pip install --no-cache-dir torch torchvision pytorch-lightning

# Expose Jupyter port (if needed)
EXPOSE 8888

# Allow argument passing
ENTRYPOINT ["conda", "run", "-n", "depth-jitter", "python", "train_q2l.py"]
