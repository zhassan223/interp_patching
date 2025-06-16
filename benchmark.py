import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import pandas as pd
import json
from typing import Dict
class WordCountingBenchmark:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
    
    def extract_answer(self, generated_text: str) -> int:
        """Extract numerical answer from generated text."""
        # Look for "Answer : (number)" format specifically
        match = re.search(r'Answer\s*:\s*\((\d+)\)', generated_text)
        if match:
            return int(match.group(1))
        else: 
            #if no answer and it decides to just igve parantheses (10) 
            match = re.search(r'\((\d+)\)', generated_text)
            if match:
                return int(match.group(1))
        #try another thing which I've observed
        
        return -1  # Invalid response
    
    def evaluate_model(self, model_name: str, max_examples: int = 1000) -> Dict:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        results = []
        correct = 0
        #gpu restrictions lol 
        UPDATED_EXAMPLES = 200
        for i, example in enumerate(tqdm(self.dataset[:UPDATED_EXAMPLES])):
            prompt = example['prompt']
            correct_answer = example['correct_answer']
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print("this is the generated_text",generated_text)
            predicted_answer = self.extract_answer(generated_text)
            
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct += 1
            
            results.append({
                'example_id': example['id'],
                'predicted': predicted_answer,
                'correct': correct_answer,
                'is_correct': is_correct,
                'list_length': example['list_length'],
                'category': example['target_category'],
                'generated_text': generated_text
            })
        
        accuracy = correct / len(results)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'total_examples': len(results),
            'correct_predictions': correct,
            'results': results,
            'generated_text':generated_text

        }
# generated list of variable AI 
models_to_test = [
     "Qwen/Qwen2.5-0.5B-Instruct",              
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",  
    "microsoft/Phi-4-mini-instruct",         
]


benchmark = WordCountingBenchmark('word_counting_dataset.json')
benchmark_results = []

for model_name in models_to_test:
    try:
        result = benchmark.evaluate_model(model_name, max_examples=500)
        benchmark_results.append(result)
        print(f"{model_name}: {result['accuracy']:.3f} accuracy")
    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Save results
with open('benchmark_results.json', 'w') as f:
    json.dump(benchmark_results, f, indent=2)