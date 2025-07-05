import random
from typing import Tuple, Optional
from datasets import Dataset


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

PRE_PROMPT = """You will be given a debate topic and your position (PRO or CON). You should reason carefully about the position, then provide your argument.
It is very important that you put your reasoning process inside <reasoning> tags and your final argument inside <answer> tags, like this:

<reasoning>
Your step-by-step reasoning process here, considering key points and potential counterarguments
</reasoning>
<answer>
Your clear, concise 2-3 sentence debate position
</answer>

All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>. 
"""

JUDGE_PROMPT = """You are an impartial debate judge. You will be shown two debate responses on the same topic, arguing the same side (PRO or CON). Your task is to determine which argument was more compelling based on:
1. Logical reasoning and evidence
2. Clear structure and organization
3. Effective use of examples
4. Respectful tone
5. Addressing potential counterarguments

Topic: {topic}

Argument 1:
{arg1_response}

Argument 2:
{arg2_response}

Which response was more compelling? Respond with EXACTLY one of these options:
- ARGUMENT_1_WINS
- ARGUMENT_2_WINS

YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED!
"""


def create_debate_examples(topics: list[str]) -> list[dict]:
    """
    Create debate examples from topics with random position assignments.
    
    Args:
        topics: List of debate topics
        
    Returns:
        List of dictionaries with 'question' and 'position' keys
    """
    examples = []
    for topic in topics:
        for position in ["PRO", "CON"]:
            formatted_question = f"Debate Topic: {topic}\nPosition: {position}"
            non_position = "CON" if position == "PRO" else "PRO"
            formatted_question_opponent = f"Debate Topic: {topic}\nPosition: {non_position}"
            examples.append({
                'question': formatted_question,
                'position': position,
                'topic': topic,
                'pre_prompt': PRE_PROMPT,
                'system_prompt': SYSTEM_PROMPT,
                "prompt": [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": PRE_PROMPT},
                    {"role": "user", "content": formatted_question},
                ],
                "prompt_opponent": [
                    {"role": "system", "content": PRE_PROMPT},
                    {"role": "user", "content": formatted_question_opponent},
                ],
                "judge_prompt_template": [
                    {"role": "system", "content": "You are an impartial debate judge."},
                    {"role": "user", "content": JUDGE_PROMPT},
                ]
            })
    
    return examples


def evaluate_judge_response(response: str) -> Optional[int]:
    """
    Evaluate the judge's response to determine if it is a win or loss.
    """
    if "ARGUMENT_1_WINS" in response:
        return True
    elif "ARGUMENT_2_WINS" in response:
        return False
    else:
        return None


def build_debate_hf_datasets(test_size: float = 0.16) -> Tuple[Dataset, Dataset]:
    """
    Build Hugging Face datasets for debate topics.
    By default, 84 train examples and 16 test examples are created.
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Define debate topics - non-controversial but engaging topics
    topics = [
        "Video games should be taught as a school sport",
        "All schools should have mandatory cooking classes",
        "Homework should be replaced with project-based learning",
        "Every city should have a night market",
        "Movie theaters should have special quiet showings",
        "All schools should teach sign language",
        "Restaurants should offer smaller portion options",
        "Public spaces should have musical instruments",
        "All high schools should start after 9am",
        "Zoos should focus only on local wildlife",
        "Libraries should have recording studios",
        "Every workplace should allow pets",
        "Schools should teach financial literacy",
        "All restaurants should show calorie counts",
        "Museums should be open late on weekends",
        "Cities should have designated graffiti walls",
        "Schools should teach basic coding",
        "Grocery stores should have recipe stations",
        "All buildings should have rooftop gardens",
        "Cafes should have board game nights",
        "Libraries should offer virtual reality rooms",
        "Parks should have outdoor movie screens",
        "Schools should teach meditation",
        "Restaurants should compost food waste",
        "Cities should have more water fountains",
        "All schools should have maker spaces",
        "Gyms should offer childcare",
        "Libraries should loan art pieces",
        "Hotels should adopt shelter pets",
        "Schools should teach gardening",
        "Airports should have sleeping pods",
        "Malls should have indoor gardens",
        "Restaurants should grow their own herbs",
        "Cities should have free music venues",
        "Schools should teach public speaking",
        "Offices should have nap rooms",
        "Supermarkets should have tasting stations",
        "Libraries should have podcast studios",
        "Parks should have outdoor chess tables",
        "Schools should teach time management",
        "Restaurants should offer cooking classes",
        "Cities should have stargazing areas",
        "Beaches should have free sunscreen",
        "Schools should teach digital citizenship",
        "Hotels should have community spaces",
        "Parks should have fruit trees",
        "Libraries should offer language exchanges",
        "Theaters should have subtitle options",
        "Schools should teach environmental science",
        "Cities should have interactive art installations"
    ]
    
    # Split into train/test sets (85/15 split)
    total_topics = len(topics)
    test_size = int(total_topics * test_size)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_topics), test_size)
    test_indices_set = set(test_indices)
    
    # Split topics
    train_topics = [t for i, t in enumerate(topics) if i not in test_indices_set]
    test_topics = [topics[i] for i in test_indices]
    
    # Create examples
    train_examples = create_debate_examples(train_topics)
    test_examples = create_debate_examples(test_topics)
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_examples)
    test_dataset = Dataset.from_list(test_examples)
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = build_debate_hf_datasets()
    print(train_dataset.shape)
    print(test_dataset.shape)
