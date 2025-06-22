# Fenrir CLI Agent

A command-line agent that converts natural language instructions into executable shell commands using a fine-tuned Gemma-3 model.

## Overview

Fenrir takes natural language instructions and generates corresponding shell commands using a fine-tuned language model. The system includes benchmarking capabilities to evaluate model performance and data collection tools for training dataset generation.

## Features

- Natural language to shell command translation
- Fine-tuned Gemma-3 model with LoRA adapters
- Comprehensive benchmarking system
- Automated data collection for training
- Detailed execution logging
- Support for both Windows and Unix-like systems

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for model inference

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fenrir
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install torch transformers peft bitsandbytes
pip install nltk rouge-score pandas
pip install python-dotenv browser-use langchain-google-genai
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

## Model Setup

The project requires a fine-tuned Gemma-3 model with LoRA adapters. The adapters should be placed in the `gemma-3/` directory.

Required files in `gemma-3/`:
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer_config.json`
- `tokenizer.json`
- `tokenizer.model`

## Usage

### Basic Command Generation

Run the agent with a natural language instruction:

```bash
python agent.py "Create a new Git branch and switch to it"
```

Example output:
```
=== CLI Agent Starting ===
Instruction: Create a new Git branch and switch to it
Loading fine-tuned model...
Fine-tuned model loaded successfully

Generating step-by-step plan...
Raw model response: git checkout -b new-branch

=== Processing instruction: Create a new Git branch and switch to it ===
Generated 1 commands:

Step 1: git checkout -b new-branch
  → git checkout -b new-branch

=== Task completed. 1 commands processed ===
Check logs/trace.jsonl for detailed execution log.
```

### Benchmarking

Evaluate model performance against test cases:

```bash
python benchmark.py
```

This will:
- Load both base and fine-tuned models
- Run test prompts through both models
- Calculate BLEU and ROUGE scores
- Generate a detailed performance report
- Save results to `benchmark_results_YYYYMMDD_HHMMSS.json`

### Data Collection

Generate training data for model improvement:

```bash
python data_scraper.py
```

Requires `GOOGLE_API_KEY` in environment variables or `.env` file.

## Project Structure

```
fenrir/
├── agent.py                    # Main CLI agent
├── benchmark.py                # Model benchmarking system
├── data_scraper.py            # Training data collection
├── gemma-3/                   # Fine-tuned model adapters
├── logs/                      # Execution logs
├── .env.example              # Environment variable template
└── README.md                 # This file
```

## Configuration

Create a `.env` file for API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
```

## Logging

All command executions are logged to `logs/trace.jsonl` with:
- Timestamp
- Original instruction
- Generated commands
- Execution results
- Error information

## Supported Command Types

The agent can generate commands for:
- Git operations
- File system manipulation
- Python environment management
- Data processing
- System administration
- Network operations

## Example Instructions

```bash
python agent.py "Compress the folder reports into reports.tar.gz"
python agent.py "List all Python files in the current directory recursively"
python agent.py "Set up a virtual environment and install requests"
python agent.py "Fetch only the first ten lines of a file named output.log"
```

## Benchmarking Results

The benchmark system evaluates:
- Command accuracy
- BLEU scores for text similarity
- ROUGE-L scores for sequence overlap
- Processing time
- Success rate

Results are saved with timestamps for tracking improvements.

## Safety

The agent runs in dry-run mode by default, echoing commands without executing them. For actual execution, manual verification of generated commands is recommended.

## Contributing

1. Ensure all tests pass with `python benchmark.py`
2. Add test cases for new functionality
3. Update documentation for new features
4. Follow existing code style and structure

## License

[Add your license information here]

## Troubleshooting

### Model Loading Issues
- Ensure `gemma-3/` directory contains all required adapter files
- Check GPU memory availability
- Verify CUDA installation if using GPU

### Dependency Issues
- Update pip: `python -m pip install --upgrade pip`
- Install specific versions if conflicts occur
- Use virtual environment to isolate dependencies

### Performance Issues
- Monitor GPU memory usage
- Adjust batch sizes if memory constrained
- Consider CPU inference for smaller models
