# Overlap Density: A Data-Centric Approach to Weak-to-Strong Generalization (W2SG)

This repository provides a research toolkit for exploring Overlap Density — a data-centric feature that has shown potential for improving Weak-to-Strong Generalization. This work builds on the groundbreaking research by Shin et al. (2024), “Weak-to-Strong Generalization Through the Data-Centric Lens.” The toolset aims to help researchers:

- Investigate the role of Overlap Density in W2SG between LLMs.
- Gain insights into how Overlap Density promotes stronger generalization from weaker models.
- Reproduce or build upon experimental results related to Overlap Density.

You can read my analysis on the Overlap Density from the research paper on LessWrong: **[The Overlap Paradigm: Rethinking Data's Role in Weak-to-Strong Generalization (W2SG)](https://www.lesswrong.com/posts/NDRfpD3q4EJ46z54Y/the-overlap-paradigm-rethinking-data-s-role-in-weak-to)**

This post summarizes my capstone project for the AI Alignment course by BlueDot Impact. You can learn more about their amazing courses [here](https://bluedot.org) and consider applying!

-------------------------------------------------------------------------------
## Table of Contents

1. [Background](#background)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Example](#example)  
6. [Contributing](#contributing)  
7. [License](#license)  
8. [Citation](#citation)  
9. [FAQ](#faq)  

-------------------------------------------------------------------------------
## Background

Traditional machine learning approaches often rely on large-scale models and vast datasets to force better performance. However, recent work (Shin et al., 2024) suggests that carefully curated data features—such as Overlap Density—can significantly impact a model’s ability to go from weak performance to strong performance. By focusing on Overlap Density, this repository provides a framework for researchers to:

- Measure and analyze Overlap Density in various datasets.  
- Explore how Overlap Density impacts model generalization.  
- Compare different modeling approaches under varying levels of Overlap Density.

-------------------------------------------------------------------------------
## Features

The notebook ([overlap_density.ipynb](notebooks/overlap_density.ipynb)) walks through the following key steps:

1. **Dataset Loading and Processing**  
   Loads and processes your choice of dataset, creating splits for training, validation, and testing.

2. **Model Initialization**  
   Initializes one or more models—optionally with LoRA (Low-Rank Adaptation)—configured within the notebook.

3. **Activation Collection**  
   Collects activations for different dataset splits and saves these details to disk.

4. **Overlap Density Calculation**  
   Computes overlap density by analyzing the collected activations alongside labels; includes change point detection logic to determine threshold boundaries.

5. **Mixing Experiments**  
   Executes experiments that mix overlapping and non-overlapping data points, examining how different proportions affect model performance.

6. **Results Visualization**  
   Uses Matplotlib (and potentially other libraries) to visualize accuracy vs. the proportion of Overlap Density.

7. **Results Saving**  
   Outputs experiment results and metrics to a JSON file for later analysis or publication.

-------------------------------------------------------------------------------
## Installation

For my experiments, I've used RunPod.io, more specifically — a docker image via their platform: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04.
Nevertheless, the code should work on any system with compatible GPU, updated drivers and required libraries installed.

Below is a recommended setup process:

1. **Clone the Repository:**
```bash
git clone https://github.com/SergioDreamer/overlap_density_w2sg.git
cd overlap_density_w2sg
```

2. **Create and Activate a Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate
# On Windows, use:
# venv\Scripts\activate
```

3. **Install the CUDA Toolkit:**
   - Make sure you have a compatible CUDA toolkit installed; for an RTX 3090, CUDA 12.4 is recommended (or a version compatible with your GPU).  
   - Follow the official [NVIDIA CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads) for your operating system.

4. **Install PyTorch and torchvision:**
   - Run the following command to install PyTorch 2.4.1 and torchvision 0.19.1 with CUDA 12.4 support (adjust if your environment requires different versions):
```bash
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

5. **Install Unsloth:**
   - This library depends on specific CUDA and PyTorch versions. Use:
```bash
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

6. **Install Additional Dependencies:**
```bash
pip install -r requirements.txt
```

-------------------------------------------------------------------------------
## Usage

Once dependencies are installed, you can open the main notebook in Jupyter or any other environment of your choice. For example:

```bash
jupyter notebook notebooks/overlap_density.ipynb
```

Follow the notebook cells in order:
1. Setup environment and libraries.  
2. Load and process your dataset.  
3. Initialize and train models for your chosen configuration.  
4. Collect activations and compute Overlap Density.  
5. Perform mixing experiments and visualize results.

-------------------------------------------------------------------------------
## Example

If you want a quick demonstration, you can run through the [overlap_density.ipynb](notebooks/overlap_density.ipynb) step by step. It includes:

• Environment checks and setup.  
• Data loading from a sample dataset.  
• Training snippet for demonstration.  
• Overlap Density calculation and threshold determination.  
• Visualization and JSON-based result logging.

-------------------------------------------------------------------------------
## Contributing

Contributions are welcome! Feel free to open issues for bugs, questions, or feature requests. To contribute via pull requests:

1. Fork the repository.  
2. Create a new branch (`git checkout -b feature/my-feature`).  
3. Make your changes and commit (`git commit -m "Add new feature"`).  
4. Push to your branch (`git push origin feature/my-feature`).  
5. Create a pull request describing your changes.

-------------------------------------------------------------------------------
## License

Please refer to the [LICENSE](LICENSE) file for license details.

-------------------------------------------------------------------------------
## Citation

If you use this toolkit in your research, please consider citing both this repository and the paper by Shin et al. (2024). A typical citation might look like:

```
@article{shin2024weak,
  title={Weak-to-Strong Generalization Through the Data-Centric Lens},
  author={Shin, Changho and Cooper, John and Sala, Frederic},
  journal={arXiv preprint arXiv:2412.03881},
  year={2024}
}

@misc{sergiodreamer_overlap_density,
  title={Research Toolkit for Overlap Density},
  author={{SergioDreamer and contributors}},
  howpublished={\url{https://github.com/SergioDreamer/overlap_density_w2sg}},
  year={2025}
}
```

-------------------------------------------------------------------------------
## FAQ

1. **Which GPU should I use?**  
   Any GPU capable of supporting the required CUDA and PyTorch versions should work. If you have an RTX 3090, CUDA 12.4 is recommended.

2. **Do I need a specific dataset?**  
   You can adapt the notebook to a variety of datasets, as long as they are in a compatible format for your data loading utilities.
