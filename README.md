# Vector Post-Training Quantization (VPTQ) for LLMs

Vector Post-Training Quantization (VPTQ) is an advanced technique designed to compress large language models (LLMs) to ultra-low bit representations (down to 2 bits) without significant performance loss. This method allows for efficient use of models in resource-constrained environments, reducing memory requirements, optimizing storage, and decreasing memory bandwidth load during inference.

## Preparing to run the model

To successfully run the model on a GPU using VPTQ, you need to ensure that all system requirements and necessary components are installed.

### Requirements

1. **Operating System**: The project has been tested and works on **Linux**. Operation on Windows has not been verified.
2. **CUDA**: Version **CUDA 12.1** (compatible with PyTorch).
3. **GPU**: An NVIDIA GPU with CUDA support is required.

### Step 1: Installing and verifying CUDA

Before starting the installation, make sure that CUDA is installed and configured. Run the command:

```bash
nvcc --version
```

This will show the version of CUDA installed on your system. If the version does not match **CUDA 12.1**, download and install the correct version from the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) website.

After installation, add environment variables for proper CUDA operation. Add the following lines to your shell configuration file (e.g., `.bashrc` or `.zshrc`):

```bash
export CUDA_PATH=/usr/local/cuda-12.1
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

Apply the changes:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Step 2: Installing and setting up a virtual environment

To manage dependencies, create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Update pip to the latest version:

```bash
pip install --upgrade pip
```

### Step 3: Installing VPTQ

Download the latest version of VPTQ .whl from [Releases](https://github.com/microsoft/VPTQ/releases) and install it by running the command:

```bash

pip install path/to/VPTQ-version-cp3xx-cp3xx-linux_x86_64.whl

```

Replace 3xx with your Python version (e.g., 39, 310, 311, or 312).

Note: Replace path/to/ with the actual path to the downloaded .whl file, and version with the corresponding version.

Required dependencies such as torch, transformers, accelerate, and datasets will be installed automatically when installing VPTQ.

### Step 4: Running the user interface

Run the `UI.py` script to open the web interface via Gradio:

```bash
python UI.py
```

This will start a web server accessible at `http://0.0.0.0:7860`. Open this URL in your browser to interact with the model. Enter a query in the text field and adjust the maximum length of the generated text using the slider.

### API Usage Example

Additionally, you can integrate VPTQ directly into your Python applications using the provided API. Example:

```python
import vptq
import transformers

# Load the tokenizer and quantized model
tokenizer = transformers.AutoTokenizer.from_pretrained("VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft")
model = vptq.AutoModelForCausalLM.from_pretrained("VPTQ-community/Meta-Llama-3.1-70B-Instruct-v8-k65536-0-woft", device_map='auto')

# Prepare the input
inputs = tokenizer("Explain: Do Not Go Gentle into That Good Night", return_tensors="pt").to("cuda")

# Generate output
outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Troubleshooting

- **CUDA issues**: If you encounter errors related to CUDA, make sure it is correctly installed and environment variables are set properly. Verify the CUDA installation using the `nvcc --version` command and ensure that the version matches the one required for PyTorch.

- **Model loading errors**: Make sure the model name in `UI.py` is specified correctly and the model files are available locally. If the model fails to load from the cache, the script will attempt to download it. This requires a stable internet connection.

- **Port conflicts**: If port `7860` is already in use, you can change the `server_port` parameter in the `iface.launch` method inside `UI.py`, selecting an available port.

- **Required dependencies not found**: If for some reason the dependencies were not installed automatically, repeat the installation manually by running the command:

   ```bash
   pip install torch>=2.2.0 transformers>=4.44.0 accelerate>=0.33.0 datasets
   ```
