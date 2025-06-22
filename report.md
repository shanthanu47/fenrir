# Fenrir CLI Agent - Project Report

## Overview

Fenrir is a command-line agent that converts natural language instructions into executable shell commands using a fine-tuned Gemma-3 language model. The project evaluates the effectiveness of model fine-tuning for improving command generation accuracy.

## System Architecture

Core Components:
- agent.py - Main CLI interface for natural language to command translation
- benchmark.py - Performance evaluation system comparing base vs fine-tuned models
- data_scraper.py - Training data collection tool
- gemma-3/ - Fine-tuned model adapters and configuration

Model Setup:
- Base Model: Gemma-3-1B with 4-bit quantization
- Fine-tuning: LoRA adapters for parameter-efficient training
- Input: Natural language instructions
- Output: Structured shell command sequences

## Performance Results

Static Evaluation (Base vs Fine-tuned):
- BLEU Score: Base 0.019 vs Fine-tuned 0.018 (minimal difference)
- ROUGE Score: Base 0.169 vs Fine-tuned 0.170 (slight improvement)
- Quality Score: Both models averaged 1.14/2 (identical performance)

Dynamic Evaluation (Agent Runs):
- Overall Success Rate: 90% (18/20 points)
- Perfect Scores: 8 out of 10 test runs
- Partial Scores: 2 out of 10 test runs
- Failed Scores: 0 out of 10 test runs

Task Performance:
- Git Operations: Excellent accuracy in branch creation and management
- File Operations: Strong performance in compression, search, and permissions
- Environment Setup: Reliable virtual environment and package installation
- System Administration: Good process monitoring and logging capabilities

## Key Findings

Strengths:
- High success rate on common CLI operations (90%)
- Accurate multi-step command sequences
- Good cross-platform command understanding
- Consistent performance across different task categories

Areas for Improvement:
- Command flag completeness (missing optional but recommended flags)
- Some responses focus on explanation rather than executable commands
- Mixed results between base and fine-tuned models suggest need for better training data

## Technical Implementation

Training Approach:
- 4-bit quantization for memory efficiency
- LoRA adapters for targeted fine-tuning
- Parameter-efficient training on command-line datasets
- CUDA acceleration for inference

Evaluation Framework:
- Automated benchmarking with 7 core test scenarios
- Multi-metric assessment using BLEU, ROUGE, and quality scoring
- 0-2 point scoring system for agent run evaluation
- Comparative analysis between base and fine-tuned models

## Conclusion

The Fenrir CLI agent demonstrates strong performance in converting natural language to shell commands with a 90% success rate in dynamic testing. While fine-tuning showed mixed results in static evaluation, the overall system proves effective for common command-line operations. Future improvements should focus on training data quality and command flag completeness.
