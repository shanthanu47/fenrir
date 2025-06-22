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

nltk.download('punkt')

class ModelBenchmark:
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.finetuned_model = None
        self.finetuned_tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        self.test_prompts = [
            "Create a new Git branch and switch to it.",
            "Compress the folder reports into reports.tar.gz.",
            "List all Python files in the current directory recursively.",
            "Set up a virtual environment and install requests.",
            "Fetch only the first ten lines of a file named output.log.",
            "Set up a complete Python development environment with virtual environment, install dependencies from requirements.txt, initialize git repository, and create initial commit.",
            "Find all processes consuming more than 80% CPU, log their details to a file, and send a notification email to admin@company.com."
        ]
        
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
        model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
        
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
        
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        return True

    def load_finetuned_model(self):
        if not os.path.exists("gemma-3"):
            return False
            
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
        
        self.finetuned_model = PeftModel.from_pretrained(base_model, "gemma-3")
        
        if self.finetuned_tokenizer.pad_token is None:
            self.finetuned_tokenizer.pad_token = self.finetuned_tokenizer.eos_token
            
        return True

    def generate_response(self, model, tokenizer, prompt: str) -> str:
        system_prompt = "You are an expert at using the command line. Given the following instruction, provide a sequence of shell commands to accomplish the task. The commands should be separated by commas and should be practical and executable.\n\nInstruction: "
        
        full_prompt = system_prompt + prompt + "\n\nCommands:"
        
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if full_prompt in response:
            response = response[len(full_prompt):].strip()
        
        response = response.replace('\n', ' ').strip()
        
        return response

    def get_fallback_response(self, prompt: str) -> str:
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
        if ',' in response:
            commands = [cmd.strip() for cmd in response.split(',') if cmd.strip()]
        else:
            lines = response.split('\n')
            commands = []
            for line in lines:
                line = line.strip()
                if (line and not line.startswith('#') and not line.startswith('//') and 
                    any(cmd_start in line for cmd_start in ['git', 'tar', 'find', 'python', 'pip', 'head', 'powershell', 'cd', 'ls', 'mkdir', 'rm'])):
                    commands.append(line)
        
        return commands[:10]

    def calculate_bleu_score(self, reference: List[str], candidate: List[str]) -> float:
        if not reference or not candidate:
            return 0.0
        
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
        if not reference or not candidate:
            return 0.0
        
        ref_text = ' '.join(reference)
        cand_text = ' '.join(candidate)
        
        scores = self.rouge_scorer.score(ref_text, cand_text)
        return scores['rougeL'].fmeasure

    def score_plan_quality(self, commands: List[str], expected: List[str]) -> int:
        if not commands:
            return 0
        
        score = 0
        
        if commands:
            score = 1
        
        command_text = ' '.join(commands).lower()
        expected_text = ' '.join(expected).lower()
        
        key_elements = []
        for exp_cmd in expected:
            words = exp_cmd.lower().split()
            if words:
                key_elements.append(words[0])
                for word in words[1:]:
                    if word.startswith('-') or word in ['install', 'init', 'add', 'commit']:
                        key_elements.append(word)
        
        matches = sum(1 for element in key_elements if element in command_text)
        coverage = matches / len(key_elements) if key_elements else 0
        
        if coverage >= 0.7:
            score = 2
        elif coverage >= 0.3:
            score = 1
        
        return score

    def run_benchmark(self) -> Dict:
        if not self.base_model or not self.finetuned_model:
            return {}
        
        results = {
            "base_model": [],
            "finetuned_model": [],
            "comparison": []
        }
        
        for i, prompt in enumerate(self.test_prompts):
            base_response = self.generate_response(self.base_model, self.base_tokenizer, prompt)
            base_commands = self.extract_commands(base_response)
            
            finetuned_response = self.generate_response(self.finetuned_model, self.finetuned_tokenizer, prompt)
            finetuned_commands = self.extract_commands(finetuned_response)
            
            expected = self.expected_commands[i]
            
            base_bleu = self.calculate_bleu_score(expected, base_commands)
            base_rouge = self.calculate_rouge_score(expected, base_commands)
            base_quality = self.score_plan_quality(base_commands, expected)
            
            finetuned_bleu = self.calculate_bleu_score(expected, finetuned_commands)
            finetuned_rouge = self.calculate_rouge_score(expected, finetuned_commands)
            finetuned_quality = self.score_plan_quality(finetuned_commands, expected)
            
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
        
        return results

    def save_results(self, results: Dict):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def print_summary(self, results: Dict):
        if not results:
            return
        
        base_avg_bleu = sum(r["bleu_score"] for r in results["base_model"]) / len(results["base_model"])
        base_avg_rouge = sum(r["rouge_score"] for r in results["base_model"]) / len(results["base_model"])
        base_avg_quality = sum(r["quality_score"] for r in results["base_model"]) / len(results["base_model"])
        
        ft_avg_bleu = sum(r["bleu_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        ft_avg_rouge = sum(r["rouge_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        ft_avg_quality = sum(r["quality_score"] for r in results["finetuned_model"]) / len(results["finetuned_model"])
        
        ft_wins = sum(1 for comp in results["comparison"] if comp["finetuned_better"] > comp["base_better"])

def main():
    benchmark = ModelBenchmark()
    
    base_loaded = benchmark.load_base_model()
    finetuned_loaded = benchmark.load_finetuned_model()
    
    if base_loaded and finetuned_loaded:
        results = benchmark.run_benchmark()
    elif base_loaded:
        return
    else:
        return
        
    if results:
        benchmark.save_results(results)
        benchmark.print_summary(results)

if __name__ == "__main__":
    main()
