

import random
import json
from typing import List, Dict, Tuple

CATEGORIES = {
    'fruit': ['apple', 'banana', 'cherry', 'grape', 'orange', 'strawberry', 'blueberry', 'peach', 'pear', 'plum', 'mango', 'kiwi', 'watermelon', 'pineapple', 'lemon'],
    'animal': ['dog', 'cat', 'bird', 'fish', 'rabbit', 'horse', 'cow', 'pig', 'sheep', 'goat', 'lion', 'tiger', 'elephant', 'bear', 'wolf'],
    'color': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'white', 'gray', 'violet', 'indigo', 'turquoise', 'crimson'],
    'vehicle': ['car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train', 'airplane', 'boat', 'ship', 'helicopter', 'taxi', 'van', 'scooter', 'subway', 'rocket'],
}

OTHER_WORDS = ['bowl', 'lamp', 'book', 'phone', 'key', 'pen', 'paper', 'clock', 'mirror', 'window', 'door', 'wall', 'floor', 'ceiling', 'box', 'bag', 'hat', 'shoe', 'shirt', 'pants']

def generate_word_list(target_category: str, list_length: int, target_count: int) -> List[str]:
    if target_count > list_length:
        raise ValueError("Target count cannot exceed list length")
    
    # Sample w/o replacement 
    target_words = random.sample(CATEGORIES[target_category], min(target_count, len(CATEGORIES[target_category])))
    
    # Fill remaining slots with non-target words
    remaining_slots = list_length - len(target_words)
    non_target_words = []
    
    # Mix of other category words and OTHER_WORDS
    other_categories = [cat for cat in CATEGORIES.keys() if cat != target_category]
    for _ in range(remaining_slots):
        
        if random.random() < 0.65: #65% sounds kinda good
            other_cat = random.choice(other_categories)
            non_target_words.append(random.choice(CATEGORIES[other_cat]))
        else:  # 
            non_target_words.append(random.choice(OTHER_WORDS))
    
    # Combine and shuffle
    all_words = target_words + non_target_words
    random.shuffle(all_words)
    return all_words

def create_dataset(num_examples: int = 5000) -> List[Dict]:
    """Create a dataset of word counting examples."""
    dataset = []
    categories = list(CATEGORIES.keys())
    
    for i in range(num_examples):
        # Vary list length and target count
        list_length = random.randint(5, 15)
        target_count = random.randint(0, min(list_length, 8))
        target_category = random.choice(categories)
        
        word_list = generate_word_list(target_category, list_length, target_count)
        
        '''Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [dog apple cherry bus cat grape bowl]'''
        prompt = f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_category}\nList: {word_list}\nAnswer: ()"
        
        dataset.append({
            'id': i,
            'prompt': prompt,
            'target_category': target_category,
            'word_list': word_list,
            'correct_answer': target_count,
            'list_length': list_length
        })
    
    return dataset

# Generate dataset
dataset = create_dataset(5000)

with open('word_counting_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

# print(f"Generated {len(dataset)} examples")
# print("Sample example:")
# print(dataset[0]['prompt'])
# print(f"Correct answer: {dataset[0]['correct_answer']}")
