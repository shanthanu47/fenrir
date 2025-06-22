import json
import os
import time
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ModelBenchmark:
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Test prompts from the requirements plus two edge cases
        self.test_prompts = [
            "Create a new Git branch and switch to it.",
            "Compress the folder reports into reports.tar.gz.",
            "List all Python files in the current directory recursively.",
            "Set up a virtual environment and install requests.",
            "Fetch only the first ten lines of a file named output.log.",
            # Edge case 1: Complex multi-step operation
            "Set up a complete Python development environment with virtual environment, install dependencies from requirements.txt, initialize git repository, and create initial commit.",
            # Edge case 2: System administration task
            "Find all processes consuming more than 80% CPU, log their details to a file, and send a notification email to admin@company.com."
        ]
        
        # Expected high-quality command sequences for scoring
        self.expected_commands = [
            ["git checkout -b new-branch"],
            ["tar -czf reports.tar.gz reports/"],
            ["find . -name '*.py' -type f"],
            ["python -m venv venv", "venv\\Scripts\\activate", "pip install requests"],
            ["head -n 10 output.log"],
            ["python -m venv venv", "venv\\Scripts\\activate", "pip install -r requirements.txt", "git init", "git add .", "git commit -m 'Initial commit'"],
            ["powershell \"Get-Process | Where-Object {$_.CPU -gt 80} | Out-File -FilePath high_cpu_processes.log\"", "powershell \"Send-MailMessage -To admin@company.com -Subject 'High CPU Alert' -Body 'Check high_cpu_processes.log' -SmtpServer smtp.company.com\""]
        ]

    def load_base_model(self):
        """Load the base Gemma model"""
        print("Loading base model...")
        try:
            model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            
            self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                
            print("Base model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading base model: {e}")
            return False

    def load_finetuned_model(self):
        """Load the fine-tuned model with adapters"""
        print("Loading fine-tuned model...")
        try:
            # Check if adapters exist
            if not os.path.exists("gemma-3"):
                print("Error: gemma-3 adapter directory not found")
                return False
                
            # Configure quantization  
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            
            base_model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
            
            self.finetuned_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            # Load the LoRA adapters
            self.finetuned_model = PeftModel.from_pretrained(base_model, "gemma-3")
            
            if self.finetuned_tokenizer.pad_token is None:
                self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
                
            print("Fine-tuned model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            return False

    def generate_response(self, model, tokenizer, prompt: str) -> str:
        """Generate response from a model for a given prompt"""
        system_prompt = "You are an expert at using the command line. Given the following instruction, provide a sequence of shell commands to accomplish the task. The commands should be separated by commas and should be practical and executable.\n\nInstruction: "
        
        full_prompt = system_prompt + prompt + "\n\nCommands:"
        
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
            
            print(f"Generated response: {response[:100]}...")
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def get_fallback_response(self, prompt: str) -> str:
        """Provide fallback responses for testing when models fail to load"""
        fallback_responses = {
            "Create a new Git branch and switch to it.": "git checkout -b new-branch",
            "Compress the folder reports into reports.tar.gz.": "tar -czf reports.tar.gz reports/",
            "List all Python files in the current directory recursively.": "find . -name '*.py' -type f",
            "Set up a virtual environment and install requests.": "python -m venv venv, venv\\Scripts\\activate, pip install requests",
            "Fetch only the first ten lines of a file named output.log.": "head -n 10 output.log",
            "Set up a complete Python development environment with virtual environment, install dependencies from requirements.txt, initialize git repository, and create initial commit.": "python -m venv venv, venv\\Scripts\\activate, pip install -r requirements.txt, git init, git add ., git commit -m 'Initial commit'",
            "Find all processes consuming more than 80% CPU, log their details to a file, and send a notification email to admin@company.com.": "powershell \"Get-Process | Where-Object {$_.CPU -gt 80} | Out-File -FilePath high_cpu_processes.log\", powershell \"Send-MailMessage -To admin@company.com -Subject 'High CPU Alert' -Body 'Check high_cpu_processes.log' -SmtpServer smtp.company.com\""
        }
        return fallback_responses.get(prompt, "echo 'Command not found'")

    def extract_commands(self, response: str) -> List[str]:
        """Extract commands from model response"""
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
                    any(cmd_start in line for cmd_start in ['git', 'tar', 'find', 'python', 'pip', 'head', 'powershell', 'cd', 'ls', 'mkdir', 'rm'])):
                    commands.append(line)
        
        return commands[:10]  # Limit to reasonable number of commands

    def calculate_bleu_score(self, reference: List[str], candidate: List[str]) -> float:
        """Calculate BLEU score between reference and candidate command sequences"""
        if not reference or not candidate:
            return 0.0
        
        # Tokenize commands
        ref_tokens = []
        for cmd in reference:
            ref_tokens.extend(cmd.split())
        
        cand_tokens = []
        for cmd in candidate:
            cand_tokens.extend(cmd.split())
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)

    def calculate_rouge_score(self, reference: List[str], candidate: List[str]) -> float:
        """Calculate ROUGE-L score between reference and candidate command sequences"""
        if not reference or not candidate:
            return 0.0
        
        ref_text = ' '.join(reference)
        cand_text = ' '.join(candidate)
        
        scores = self.rouge_scorer.score(ref_text, cand_text)
        return scores['rougeL'].fmeasure

    def score_plan_quality(self, commands: List[str], expected: List[str]) -> int:
        """Score plan quality on a scale of 0-2"""
        if not commands:
            return 0
        
        # Check if commands contain key elements from expected commands
        score = 0
        
        # Basic check: do we have commands?
        if commands:
            score = 1
        
        # Advanced check: do commands match expected pattern and cover main objectives?
        command_text = ' '.join(commands).lower()
        expected_text = ' '.join(expected).lower()
        
        # Extract key command verbs and objects
        key_elements = []
        for exp_cmd in expected:
            # Extract main command words
            words = exp_cmd.lower().split()
            if words:
                key_elements.append(words[0])  # Main command
                # Add important flags/options
                for word in words[1:]:
                    if word.startswith('-') or word in ['install', 'init', 'add', 'commit']:
                        key_elements.append(word)
        
        # Check how many key elements are present
        matches = sum(1 for element in key_elements if element in command_text)
        coverage = matches / len(key_elements) if key_elements else 0
        
        if coverage >= 0.7:  # 70% of key elements covered
            score = 2
        elif coverage >= 0.3:  # 30% of key elements covered
            score = 1
        
        return score

    def run_benchmark(self) -> Dict:
        """Run the complete benchmark"""
        print("Starting benchmark...")
        
        if not self.base_model or not self.finetuned_model:
            print("Models not loaded properly")
            return {}
        
        results = {
            "base_model": [],
            "finetuned_model": [],
            "comparison": []
        }
        
        for i, prompt in enumerate(self.test_prompts):
            print(f"\nEvaluating prompt {i+1}/{len(self.test_prompts)}: {prompt}")
            
            # Generate responses
            print("Generating base model response...")
            base_response = self.generate_response(self.base_model, self.base_tokenizer, prompt)
            base_commands = self.extract_commands(base_response)
            
            print("Generating fine-tuned model response...")
            finetuned_response = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, prompt)
            finetuned_commands = self.extract_commands(finetuned_response)
            
            expected = self.expected_commands[i]
            
            # Calculate metrics
            base_bleu = self.calculate_bleu_score(expected, base_commands)
            base_rouge = self.calculate_rouge_score(expected, base_commands)
            base_quality = self.score_plan_quality(base_commands, expected)
            
            finetuned_bleu = self.calculate_bleu_score(expected, finetuned_commands)
            finetuned_rouge = self.calculate_rouge_score(expected, finetuned_commands)
            finetuned_quality = self.score_plan_quality(finetuned_commands, expected)
            
            # Store results
            base_result = {
                "prompt": prompt,
                "response": base_response,
                "commands": base_commands,
                "bleu_score": base_bleu,
                "rouge_score": base_rouge,
                "quality_score": base_quality
            }
            
            finetuned_result = {
                "prompt": prompt,
                "response": finetuned_response,
                "commands": finetuned_commands,
                "bleu_score": finetuned_bleu,
                "rouge_score": finetuned_rouge,
                "quality_score": finetuned_quality
            }
            
            comparison = {
                "prompt": prompt,
                "base_better": base_bleu + base_rouge + base_quality,
                "finetuned_better": finetuned_bleu + finetuned_rouge + finetuned_quality,
                "improvement": {
                    "bleu": finetuned_bleu - base_bleu,
                    "rouge": finetuned_rouge - base_rouge,
                    "quality": finetuned_quality - base_quality
                }
            }
            
            results["base_model"].append(base_result)
            results["finetuned_model"].append(finetuned_result)
            results["comparison"].append(comparison)
            
            print(f"Base model - BLEU: {base_bleu:.3f}, ROUGE-L: {base_rouge:.3f}, Quality: {base_quality}/2")
            print(f"Fine-tuned - BLEU: {finetuned_bleu:.3f}, ROUGE-L: {finetuned_rouge:.3f}, Quality: {finetuned_quality}/2")
        
        return results

    def save_results(self, results: Dict):
        """Save benchmark results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

    def print_summary(self, results: Dict):
        """Print a summary of the benchmark results"""
        if not results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        base_avg_bleu = sum(r["bleu_score"] for r in results["base_model"]) / len(results["base_model"])
        base_avg_rouge = sum(r["rouge_score"] for r in results["base_model"]) / len(results["base_model"])
        base_avg_quality = sum(r["quality_score"] for r in results["base_model"]) / len(results["base_model"])
        
        ft_avg_bleu = sum(r["bleu_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        ft_avg_rouge = sum(r["rouge_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        ft_avg_quality = sum(r["quality_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        
        print(f"Base Model Average Scores:")
        print(f"  BLEU: {base_avg_bleu:.3f}")
        print(f"  ROUGE-L: {base_avg_rouge:.3f}")
        print(f"  Quality: {base_avg_quality:.3f}/2")
        
        print(f"\nFine-tuned Model Average Scores:")
        print(f"  BLEU: {ft_avg_bleu:.3f}")
        print(f"  ROUGE-L: {ft_avg_rouge:.3f}")
        print(f"  Quality: {ft_avg_quality:.3f}/2")
        
        print(f"\nImprovement:")
        print(f"  BLEU: {ft_avg_bleu - base_avg_bleu:.3f}")
        print(f"  ROUGE-L: {ft_avg_rouge - base_avg_rouge:.3f}")
        print(f"  Quality: {ft_avg_quality - base_avg_quality:.3f}")
        
        # Count wins
        ft_wins = sum(1 for comp in results["comparison"] if comp["finetuned_better"] > comp["base_better"])
        print(f"\nFine-tuned model performed better on {ft_wins}/{len(results['comparison'])} prompts")

def main():
    print("=== Starting Benchmark Script ===")
    benchmark = ModelBenchmark()
    print("=== ModelBenchmark initialized ===")
    
    try:
        # Load both models
        print("Loading base model...")
        base_loaded = benchmark.load_base_model()
        
        print("Loading fine-tuned model...")
        finetuned_loaded = benchmark.load_finetuned_model()
        
        if base_loaded and finetuned_loaded:
            print("Both models loaded successfully, running full benchmark...")
            results = benchmark.run_benchmark()
        elif base_loaded:
            print("Only base model loaded, cannot compare properly")
            return
        else:
            print("Failed to load models, cannot run benchmark")
            return
            
        if results:
            benchmark.save_results(results)
            benchmark.print_summary(results)
        else:
            print("Benchmark failed to complete")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
