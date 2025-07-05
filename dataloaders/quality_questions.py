from datasets import load_dataset, Dataset, DatasetDict


DEBATER_PROMPT = """\
You are an expert debater number `{debater_id}` engaging in a debate over answers to reading comprehension questions. You are going to debate with your interlocutor on this question : '{question}'
Your stance in the debate is: '{pov}'
Your interlocutor's stance is: '{interlocutor_pov}'
Here is the reading comprehension article: '{article}'
Don't precise the role you're going to play, start directly the debate with your oponent, and defend the your stance. Do not exceed 300 words per answer. A good rule of thumb is that a short paragraph and one quote take approximately 100 words.
Give your answer:
"""
JUDGE_PROMPT_NO_DEBATE_TEXT = """\
You are judging a debate between two persons on this question : '{question}'

Point of view of debater 1 is : '{answer_1}'

Point of view of debater 2 is : '{answer_2}'

Which point of view seems right ? Respond by typing '1' or '2'.
"""
JUDGE_PROMPT_WITH_DEBATE_TEXT = """\
You are judging a debate between two persons on this question : '{question}'

Point of view of debater 1 is : '{answer_1}'

Point of view of debater 2 is : '{answer_2}'

Here is the debate: '{debate_text}'

Given the arguments of both debaters, which point of view seems right ? Respond by typing '1' or '2'.
"""


class Question:
    def __init__(
        self,
        question: str,
        context: str,
        answer_1: str,
        answer_2: str,
        true_answer: int,
    ) -> None:
        self.question = question
        self.context = context
        self.answer_1 = answer_1
        self.answer_2 = answer_2
        # either idx 1 or 2 to determine true answer
        self.true_answer = true_answer


def load_quality(n_questions: int = 6) -> tuple[list[Question], list[Question]]:
    """Load and filter the QuALITY dataset, selecting unique articles with hard questions.

    Args:
        row_selected: List of indices to include in the results, if None includes all

    Returns:
        Tuple containing training and test question sets, each with 6 questions
    """
    dataset = load_dataset("emozilla/quality")

    # Process training data
    train_articles = dataset["train"]["article"]
    train_questions = dataset["train"]["question"]
    train_options = dataset["train"]["options"]
    train_answers = dataset["train"]["answer"]
    train_hard = dataset["train"]["hard"]

    # Process validation data
    val_articles = dataset["validation"]["article"]
    val_questions = dataset["validation"]["question"]
    val_options = dataset["validation"]["options"]
    val_answers = dataset["validation"]["answer"]
    val_hard = dataset["validation"]["hard"]

    # Function to process and get questions from a dataset split
    def get_questions_from_split(articles, questions, options, answers, hard, count=n_questions):
        # Group indices by article to ensure unique articles
        article_to_indices = {}
        for i in range(len(articles)):
            if hard[i]:
                article = articles[i]
                if article not in article_to_indices:
                    article_to_indices[article] = []
                article_to_indices[article].append(i)

        # Sort articles by length and select the first 'count' articles
        sorted_articles = sorted(article_to_indices.keys(), key=len)

        # Take only up to 'count' articles to ensure uniqueness
        selected_articles = sorted_articles[
            : count * 2
        ]  # Take more initially in case we need to skip some

        # Take one question from each unique article, skipping problematic answers
        selected_indices = []
        for article in selected_articles:
            # Try to find a suitable question for this article
            valid_question_found = False

            for idx in article_to_indices[article]:
                option_list = options[idx]

                # Check if any answer option contains phrases like "All of the options are correct"
                invalid_phrases = ["all of the", "all the", "both are", "none of the"]
                has_invalid_answer = any(
                    any(phrase in option.lower() for phrase in invalid_phrases)
                    for option in option_list
                )

                if not has_invalid_answer:
                    selected_indices.append(idx)
                    valid_question_found = True
                    break

            # If we found a valid question and have enough, stop
            if valid_question_found and len(selected_indices) >= count:
                break

        # Take just the first 'count' indices
        selected_indices = selected_indices[:count]

        # If we don't have enough articles with valid questions, duplicate some questions
        while len(selected_indices) < count:
            selected_indices.append(selected_indices[0] if selected_indices else 0)

        # Create Question objects
        questions_list = []
        for i in selected_indices:
            article = articles[i]
            question_text = questions[i]
            option_list = options[i]
            correct_answer_idx = answers[i]

            # Get the correct and incorrect answers
            answer_1 = option_list[correct_answer_idx]
            answer_2 = option_list[1 - correct_answer_idx]

            # Create question object
            question = Question(question_text, article, answer_1, answer_2, 1)
            questions_list.append(question)

        return questions_list

    # Get training questions from train split
    train_questions_list = get_questions_from_split(
        train_articles, train_questions, train_options, train_answers, train_hard, n_questions
    )

    # Get test questions from validation split
    test_questions_list = get_questions_from_split(
        val_articles, val_questions, val_options, val_answers, val_hard, n_questions
    )

    print(
        f"Loaded {len(train_questions_list)} training questions and {len(test_questions_list)} test questions"
    )
    return train_questions_list, test_questions_list


def questions_to_datasets(train_questions_list, test_questions_list):
    """Convert lists of Question objects back to Hugging Face datasets.
    
    Args:
        train_questions_list: List of Question objects for training
        test_questions_list: List of Question objects for testing
    
    Returns:
        DatasetDict containing 'train' and 'test' splits with fields:
        - article: The context/article text
        - questions: The question text
        - true_answer: The index (1 or 2) of the correct answer
        - answer_1: First answer option
        - answer_2: Second answer option
    """
    
    def extract_data_from_questions(questions_list):
        """Extract data from Question objects into a dictionary format."""
        data = {
            "article": [],
            "question": [],
            "true_answer": [],
            "answer_1": [],
            "answer_2": []
        }
        
        for question in questions_list:
            data["article"].append(question.context)
            data["question"].append(question.question)
            data["true_answer"].append(question.true_answer)
            data["answer_1"].append(question.answer_1)
            data["answer_2"].append(question.answer_2)
        
        return data
    
    # Extract data from both question lists
    train_data = extract_data_from_questions(train_questions_list)
    test_data = extract_data_from_questions(test_questions_list)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    print(f"Created datasets with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    return dataset_dict


def get_debater_input_message(
    question, article, debater_id, answer_1, answer_2
):
    """Generate the input message for the debater

    :param debater_id: int representing the debater id (1 or 2)
    :raises ValueError: if debater_id is not 1 or 2
    :return: list of dicts containing the input message for the debater
    """
    if debater_id == 1:
        pov = answer_1
        interlocutor_pov = answer_2
    elif debater_id == 2:
        pov = answer_2
        interlocutor_pov = answer_1
    else:
        raise ValueError(f"Invalid debater_id : {debater_id}, should be 1 or 2")

    debater_prompt = DEBATER_PROMPT.format(
        debater_id=debater_id,
        question=question,
        pov=pov,
        interlocutor_pov=interlocutor_pov,
        article=article,
    )

    message = [{"role": "user", "content": debater_prompt}]

    return message


def get_judge_input_message(debate_history_data: dict, debate_text: str):
    """Generate the input message for the judge

    :param debate_history_data: dict containing the data of the current match
    :param debate_text: str containing the debate text
    :return: list of dicts containing the input message for the judge
    """
    # if no debate text, judge only has the question and the point of views
    if not debate_text:
        judge_prompt_sentences = JUDGE_PROMPT_NO_DEBATE_TEXT.format(
            question=debate_history_data['question'],
            answer_1=debate_history_data['answer_1'],
            answer_2=debate_history_data['answer_2'],
        )
    # if debate text, judge has the full debate
    else:
        judge_prompt_sentences = JUDGE_PROMPT_WITH_DEBATE_TEXT.format(
            question=debate_history_data['question'],
            answer_1=debate_history_data['answer_1'],
            answer_2=debate_history_data['answer_2'],
            debate_text=debate_text,
        )

    message = [{"role": "user", "content": judge_prompt_sentences}]
    return message


def get_quality_questions(split: str = "train") -> Dataset:
    """Load and prepare QualityQuestions dataset."""
    train_questions, test_questions = load_quality(n_questions=6)
    dataset_dict = questions_to_datasets(train_questions, test_questions)
    data = []
    for x in dataset_dict[split]:
        for trained_position in [1, 2]:
            frozen_position = 2 if trained_position == 1 else 1
            instance = {
                'prompt': get_debater_input_message(
                    x['question'], x['article'], trained_position, x['answer_1'], x['answer_2']
                ),
                'prompt_llm_frozen': get_debater_input_message(
                    x['question'], x['article'], frozen_position, x['answer_1'], x['answer_2']
                ),
                'prompt_judge_info': {
                    'answer_1': x['answer_1'],
                    'answer_2': x['answer_2'],
                    'question': x['question'],
                },
                "prompt_judge_without_debate": get_judge_input_message(x, None),
                'answer': x['true_answer'],
                'trained_defends': trained_position
            }
            data.append(instance)

    # Convert back to dataset format if needed
    data = Dataset.from_list(data)
    return data


if __name__ == "__main__":
    data = get_quality_questions(split="train")
