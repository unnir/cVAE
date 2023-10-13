# MNIST Digit Generation with Conditional Variational Autoencoder (cVAE) using PyTorch

## Description:
Explore the power of Conditional Variational Autoencoders (CVAEs) through this implementation trained on the MNIST dataset to generate handwritten digit images based on class labels. Utilizing the robust and versatile PyTorch library, this project showcases a straightforward yet effective approach to conditional generative modeling.

## Key Features:

- Thorough implementation of CVAE as delineated in the `seminal paper`(https://arxiv.org/abs/1312.6114).
- Streamlined training and evaluation workflows for the MNIST dataset.
- Real-time visualization of generated digit images during training, alongside epoch-wise saved image outputs.
- Modular and well-commented code for ease of customization, extension, and detailed understanding.

## Table of Contents:

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Output](#output)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

## Getting Started

## Prerequisites

- `Python 3.x`(https://www.python.org/downloads/)
- `PyTorch`(https://pytorch.org/get-started/locally/)
- `Torchvision`(https://pytorch.org/vision/stable/index.html)

# Installation

1. Clone the repository:
   `bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   `

2. (Optional) It's advisable to create a virtual environment to manage dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use +env\Scripts\activate+
   ```

3. Install the necessary libraries:
   ```bash
   pip install torch torchvision
   ```

## Usage

1. Make sure the MNIST dataset is downloaded in the +../data+ directory, or modify the path in the script to your dataset location.
2. Run the main script to initiate training and evaluation of the CVAE model:
   ```bash
   python main.py
   ```

## Output

- Real-time console output of training and testing loss values.
- Generated digit images during training saved as +reconstruction_<epoch>.png+ and sampled images from the latent space saved as +sample_<epoch>.png+ in the current directory.

## Contributing

Feel free to fork the project, open issues, and submit Pull Requests. For major changes, please open an issue first to discuss what you would like to change.

## License

```MIT```(https://choosealicense.com/licenses/mit/)

## References

1. `Auto-Encoding Variational Bayes`(https://arxiv.org/abs/1312.6114) - Kingma and Welling's groundbreaking paper on Variational Autoencoders.
2. `PyTorch Documentation`(https://pytorch.org/docs/stable/index.html) - Official documentation and tutorials.
3. `MNIST dataset`(http://yann.lecun.com/exdb/mnist/) - Source of the MNIST dataset used in this project.

## Author 
Vadim Borisov
