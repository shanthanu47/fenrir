import json
import os
import subprocess
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

def load_finetuned_model():
    """Load the fine-tuned model with adapters"""
    print("Loading fine-tuned model...")
    try:
        # Check if adapters exist
        if not os.path.exists("gemma-3"):
            print("Error: gemma-3 adapter directory not found")
            return None, None
            
        # Configure quantization  
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        base_model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Load the LoRA adapters
        model = PeftModel.from_pretrained(base_model, "gemma-3")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Fine-tuned model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None, None

def generate_commands(model, tokenizer, instruction):
    """Generate shell commands from natural language instruction"""
    system_prompt = """You are an expert at using the command line. Given the following instruction, provide a sequence of shell commands to accomplish the task. The commands should be separated by commas and should be practical and executable.

Instruction: """
    
    full_prompt = system_prompt + instruction + "\n\nCommands:"
    
    try:
        # Tokenize input
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part after the prompt
        if full_prompt in response:
            response = response[len(full_prompt):].strip()
        
        # Clean up response
        response = response.replace('\n', ' ').strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

def parse_commands(response):
    """Parse the model response to extract individual commands"""
    if not response:
        return []
    
    # Try to find commands separated by commas
    if ',' in response:
        commands = [cmd.strip() for cmd in response.split(',') if cmd.strip()]
    else:
        # Try to find individual commands (look for typical command patterns)
        lines = response.split('\n')
        commands = []
        for line in lines:
            line = line.strip()
            # Look for lines that start with command-like patterns
            if (line and not line.startswith('#') and not line.startswith('//') and 
                any(cmd_start in line.lower() for cmd_start in ['git', 'tar', 'find', 'python', 'pip', 'head', 'powershell', 'cd', 'ls', 'mkdir', 'rm', 'echo', 'cat', 'grep', 'awk', 'sort'])):
                commands.append(line)
    
    # Clean up commands and remove obvious non-commands
    cleaned_commands = []
    for cmd in commands:
        cmd = cmd.strip()
        # Remove leading dashes or bullets
        cmd = re.sub(r'^[-•*]\s*', '', cmd)
        # Remove backticks
        cmd = cmd.replace('`', '')
        # Skip empty or very short commands
        if len(cmd) > 2 and not cmd.startswith('import '):
            cleaned_commands.append(cmd)
    
    return cleaned_commands[:10]  # Limit to reasonable number of commands

def execute_dry_run(commands, instruction):
    """Execute commands in dry-run mode and log each step"""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    print(f"\n=== Processing instruction: {instruction} ===")
    print(f"Generated {len(commands)} commands:")
    
    with open("logs/trace.jsonl", "a", encoding='utf-8') as log_file:
        # Log the initial instruction
        log_entry = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "step": "instruction",
            "instruction": instruction,
            "command_count": len(commands)
        }
        log_file.write(json.dumps(log_entry) + "\n")
        
        # Execute each command in dry-run mode
        for i, command in enumerate(commands, 1):
            print(f"\nStep {i}: {command}")
            
            # Log the step
            log_entry = {
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "step": "dry_run_command",
                "step_number": i,
                "command": command,
                "instruction": instruction
            }
            log_file.write(json.dumps(log_entry) + "\n")
            
            # Execute in dry-run mode (echo the command)
            try:
                result = subprocess.run(
                    f'echo "{command}"', 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8'
                )
                print(f"  → {result.stdout.strip()}")
                
                # Log the execution result
                log_entry = {
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "step": "execution_result",
                    "step_number": i,
                    "command": command,
                    "output": result.stdout.strip(),
                    "success": result.returncode == 0
                }
                log_file.write(json.dumps(log_entry) + "\n")
                
            except Exception as e:
                print(f"  → Error: {e}")
                log_entry = {
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "step": "execution_error",
                    "step_number": i,
                    "command": command,
                    "error": str(e)
                }
                log_file.write(json.dumps(log_entry) + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<natural language instruction>\"")
        print("Example: python agent.py \"Create a new Git branch and switch to it\"")
        sys.exit(1)

    instruction = sys.argv[1]
    
    print("=== CLI Agent Starting ===")
    print(f"Instruction: {instruction}")
    
    # Load the fine-tuned model
    model, tokenizer = load_finetuned_model()
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Generate command sequence
    print("\nGenerating step-by-step plan...")
    response = generate_commands(model, tokenizer, instruction)
    
    if not response:
        print("Failed to generate commands. Exiting.")
        sys.exit(1)
    
    print(f"Raw model response: {response}")
    
    # Parse commands from response
    commands = parse_commands(response)
    
    if not commands:
        print("No valid commands extracted from response.")
        sys.exit(1)
    
    # Execute commands in dry-run mode
    execute_dry_run(commands, instruction)
    
    print(f"\n=== Task completed. {len(commands)} commands processed ===")
    print("Check logs/trace.jsonl for detailed execution log.")

if __name__ == "__main__":
    main()
