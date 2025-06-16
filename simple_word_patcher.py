import torch
import json
import re
import nnsight
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import random

class WordCountPatcher:
    
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading {model_name}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = nnsight.LanguageModel(model_name, device_map='auto')
        self.num_layers = len(self.model.model.layers)
        print(f"Model loaded with {self.num_layers} layers")
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load the word counting dataset"""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples from dataset")
        return data
    
    def create_forced_prompts(self, example: Dict) -> str:
        """Create prompts that force the model to complete with a number
        
        #addition needed because the original evaluation the prompt wasn't giving in the same 
        """

        base_prompt = example['prompt']
        
        if "just give me the answer in this format: 'Answer(#)'" in base_prompt:
            base_prompt = base_prompt.replace("just give me the answer in this format: 'Answer(#)'", "")
        
        forced_prompt = base_prompt.strip() + " Answer("
        return forced_prompt
    
    def find_pairs(self, dataset: List[Dict], max_pairs: int = 50) -> List[Tuple[Dict, Dict]]:
        """Find pairs of examples with same category but different counts
        
        allows me to distinctly see the progress when patching
        """
        pairs = []
        
        # Filter out examples with low counts
        valid_examples = [ex for ex in dataset if ex.get('correct_answer', 0) > 2]
        print(f"Filtered to {len(valid_examples)} examples with count > 2")
        
        for i, example1 in enumerate(valid_examples):
            if len(pairs) >= max_pairs:
                break
                
            for j, example2 in enumerate(valid_examples[i+1:], i+1):
                if len(pairs) >= max_pairs:
                    break
                    
                # Must be same category but different counts (difference >= 2)
                if (example1.get('target_category') == example2.get('target_category') and 
                    example1.get('correct_answer', 0) != example2.get('correct_answer', 0) and
                    abs(example1.get('correct_answer', 0) - example2.get('correct_answer', 0)) >= 2):
                    
                    pairs.append((example1, example2))
        
        print(f"Found {len(pairs)} valid pairs for patching")
        return pairs
    
    def debug_forced_output(self, example: Dict):
        """Debug what the model outputs with forced prompts"""
        forced_prompt = self.create_forced_prompts(example)
        print(f"\nDEBUG: Testing FORCED prompt: {forced_prompt}")
        print(f"Expected answer: {example['correct_answer']}")
        
        with self.model.trace(forced_prompt):
            logits = self.model.lm_head.output.save()
        
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        top_indices = torch.topk(probs, 10).indices
        
        print("Top 10 predictions after 'Answer(':")
        for i, idx in enumerate(top_indices):
            token = self.tokenizer.decode([idx])
            prob = probs[idx].item()
            print(f"  {i+1}: '{token}' (prob: {prob:.4f})")
        
        number_probs = self.get_number_probabilities(logits)
        print(f"Number probabilities: {number_probs}")
        
        return number_probs
    
    def find_word_positions(self, prompt: str, word_list: List[str]) -> List[int]:
        """Find token positions where each word appears"""
        tokens = self.tokenizer.encode(prompt)
        token_strs = [self.tokenizer.decode([tok]) for tok in tokens]
        
        positions = []
        for word in word_list:
            for i, token_str in enumerate(token_strs):
                if word.lower() in token_str.lower().strip():
                    positions.append(i)
                    break
        
        return positions
    
    def get_number_probabilities(self, logits: torch.Tensor) -> Dict[int, float]:
        """Get probabilities for numbers 0-15"""
        vocab = self.tokenizer.get_vocab()
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        
        number_probs = {}
        
        for num in range(16):
            formats = [str(num), f"{num}", f" {num}"]
            
            for fmt in formats:
                if fmt in vocab:
                    token_id = vocab[fmt]
                    prob = probs[token_id].item()
                    if num not in number_probs or prob > number_probs[num]:
                        number_probs[num] = prob
        
        return number_probs
    
    def patch_layer(self, source_prompt: str, target_prompt: str, 
                   layer: int, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patch one layer at a specific position"""
        
        source_forced = self.create_forced_prompts({'prompt': source_prompt})
        target_forced = self.create_forced_prompts({'prompt': target_prompt})
        
        # Get source activation
        with self.model.trace(source_forced):
            source_activation = self.model.model.layers[layer].output[0][:, position, :].save()
        
        # Get clean target output
        with self.model.trace(target_forced):
            clean_logits = self.model.lm_head.output.save()
        
        # Patch target with source activation
        with self.model.trace(target_forced):
            self.model.model.layers[layer].output[0][:, position, :] = source_activation
            patched_logits = self.model.lm_head.output.save()
        
        return clean_logits, patched_logits
    
    def calculate_effect(self, clean_logits: torch.Tensor, patched_logits: torch.Tensor,
                        source_count: int, target_count: int) -> float:
        
        clean_probs = self.get_number_probabilities(clean_logits)
        patched_probs = self.get_number_probabilities(patched_logits)
        
        source_increase = patched_probs.get(source_count, 0) - clean_probs.get(source_count, 0)
        target_decrease = clean_probs.get(target_count, 0) - patched_probs.get(target_count, 0)
        
        effect = source_increase + target_decrease
        
        return effect
    
    def test_all_layers(self, source_example: Dict, target_example: Dict) -> Dict[int, float]:
        
        source_prompt = source_example['prompt']
        target_prompt = target_example['prompt']
        source_count = source_example['correct_answer']
        target_count = target_example['correct_answer']
        
        source_forced = self.create_forced_prompts(source_example)
        target_forced = self.create_forced_prompts(target_example)
        
        source_tokens = self.tokenizer.encode(source_forced)
        target_tokens = self.tokenizer.encode(target_forced)
        
        positions_to_try = []
        
        # Position 1: Just before "Answer(" - where final count should be computed
        pos1 = min(len(source_tokens) - 3, len(target_tokens) - 3)
        if pos1 > 0:
            positions_to_try.append(pos1)
        
        # Position 2: Middle of word list
        source_positions = self.find_word_positions(source_prompt, source_example['word_list'])
        target_positions = self.find_word_positions(target_prompt, target_example['word_list'])
        
        if source_positions and target_positions:
            mid_pos = min(source_positions[len(source_positions)//2], 
                         target_positions[len(target_positions)//2])
            positions_to_try.append(mid_pos)
        
        effects = {}
        
        for layer in range(self.num_layers):
            best_effect = 0.0
            
            for patch_position in positions_to_try:
                if patch_position < 0:
                    continue
                    
                try:
                    clean_logits, patched_logits = self.patch_layer(
                        source_prompt, target_prompt, layer, patch_position
                    )
                    
                    effect = self.calculate_effect(
                        clean_logits, patched_logits, source_count, target_count
                    )
                    
                    if abs(effect) > abs(best_effect):
                        best_effect = effect
                    
                except Exception as e:
                    continue
            
            effects[layer] = best_effect
        
        return effects
    
    def run_experiment(self, dataset_path: str, max_pairs: int = 20) -> Dict:
        """Run the full patching experiment"""
        
        dataset = self.load_dataset(dataset_path)
        
        print("\n=== DEBUGGING FORCED MODEL OUTPUTS ===")
        for i in range(min(3, len(dataset))):
            if dataset[i].get('correct_answer', 0) > 2:
                self.debug_forced_output(dataset[i])
                break
        
        pairs = self.find_pairs(dataset, max_pairs)
        
        if not pairs:
            print("No valid pairs found!")
            return {}
        
        layer_effects = {layer: [] for layer in range(self.num_layers)}
        
        print("\nTesting patching across all layers...")
        
        for i, (source_ex, target_ex) in enumerate(pairs):
            print(f"Pair {i+1}/{len(pairs)}: {source_ex['correct_answer']} → {target_ex['correct_answer']}")
            
            effects = self.test_all_layers(source_ex, target_ex)
            
            for layer, effect in effects.items():
                layer_effects[layer].append(effect)
        
        results = {}
        for layer in range(self.num_layers):
            if layer_effects[layer]:
                avg_effect = sum(layer_effects[layer]) / len(layer_effects[layer])
                results[layer] = {
                    'average_effect': avg_effect,
                    'num_tests': len(layer_effects[layer]),
                    'all_effects': layer_effects[layer]
                }
        
        return results
    
    def analyze_results(self, results: Dict):
        """Print analysis of results"""
        
        print("\n" + "="*60)
        print("PATCHING RESULTS ANALYSIS")
        print("="*60)
        
        if not results:
            print("No results to analyze!")
            return
        
        sorted_layers = sorted(results.items(), 
                             key=lambda x: abs(x[1]['average_effect']), 
                             reverse=True)
        
        print("Layers ranked by effect strength:")
        print("-" * 40)
        
        for layer, data in sorted_layers[:10]:
            avg_effect = data['average_effect']
            num_tests = data['num_tests']
            
            significance = "★★★" if abs(avg_effect) > 0.01 else "★★" if abs(avg_effect) > 0.005 else "★"
            
            print(f"Layer {layer:2d}: {avg_effect:+.6f} ({num_tests} tests) {significance}")
        
        if sorted_layers:
            best_layer, best_data = sorted_layers[0]
            print(f"\nBest layer: {best_layer}")
            print(f"Average effect: {best_data['average_effect']:+.6f}")
            
            if abs(best_data['average_effect']) > 0.001:
                print(f"This suggests layer {best_layer} contains running count representations!")
            else:
                print("No strong effects found.")
        
        return sorted_layers

def main():
    """Run the word counting patch experiment"""
    
    patcher = WordCountPatcher("Qwen/Qwen2.5-0.5B-Instruct")
    
    results = patcher.run_experiment("word_counting_dataset.json", max_pairs=30)
    
    patcher.analyze_results(results)
    
    return results

if __name__ == "__main__":
    results = main()