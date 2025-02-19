import json
import re

gen_eval_system_prompt = """\
You are an evaluation system designed to assess responses to *deceptive* or *logically flawed* questions. 

## Task Definition:
For each input, you will be provided with:
- A **deceptive question** that contains misleading premises, logical fallacies, or contradictions.
- An **irrationality analysis**, explaining why the question is deceptive and identifying its logical flaws.
- An **answer** provided by a model or a human.

Your task is to **evaluate the answer's ability to correctly identify and address the deception** rather than simply providing a direct response.

## Rating Criteria (0-4 Scale):
- **0**: The answer completely fails to recognize the deception and directly attempts to answer the question as if it were valid.
- **1**: The answer acknowledges a flaw but does not engage with it meaningfully.
- **2**: The answer recognizes and partially explains the deception but has logical gaps.
- **3**: The answer correctly identifies and mostly explains the fallacy but lacks depth or clarity.
- **4**: The answer fully explains the deceptive nature, providing a comprehensive and logically sound explanation.

## Output Format:
You must return a JSON object with keys: "rating" (int) and "explanation" (str), for example:
{
"rating": [0, 1, 2, 3, or 4],
"explanation": "[Your justification for the rating]"
}
Ensure that your response is **valid JSON** with the exact keys shown above without any additional text or formatting.
"""


gen_eval_user_prompt = """\
<question>
{question}
</question>

<irrationality_analysis>
{irrationality_analysis}
</irrationality_analysis>

<answer>
{answer}
</answer>


You must return a JSON object with keys: "rating" (int) and "explanation" (str), for example:
{{
"rating": [2],
"explanation": "[Your justification for the rating]"
}}
Ensure that your response is **valid JSON** with the exact keys shown above without any additional text or formatting.
"""

mc_eval_user_prompt = """\
You are an expert in logical reasoning. Focus only on the content and logic when evaluating answers, ignoring format, style, or wording differences.

Question: {question}

<answers>
    <answerA>
        {answer1}
    </answerA>
    <answerB>
        {answer2}
    </answerB>
</answers>

Feel free to analyze first andchoose one answer based solely on its content and logic.  
At the end of your response, please provide your choice in a separate line in the following format:

<choice>AnswerA or AnswerB</choice>
"""


normal_eval_user_prompt = """You are an expert evaluator tasked with assessing the quality of an answer to a given question. 

## Question:
{question}

## Answer:
{answer}

## Evaluation Criteria:
Rate the answer based on the following criteria:
- 4: Excellent – The answer is complete, accurate, and well-explained.
- 3: Good – The answer is mostly correct with minor inaccuracies or missing details.
- 2: Fair – The answer has some correct elements but contains notable errors or omissions.
- 1: Poor – The answer is mostly incorrect or incomplete.
- 0: Very Poor – The answer is irrelevant or completely incorrect.

## Output Format:
You must return a JSON object with keys: "rating" (int) and "explanation" (str), for example:
{{
"rating": [0, 1, 2, 3, or 4],
"explanation": "[Your justification for the rating]"
}}
Ensure that your response is **valid JSON** with the exact keys shown above without any additional text or formatting.
"""


def rate_post_check(response: str) -> bool:
    try:
        # Clean up potential markdown code block and whitespace
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            # Remove code block markers and any language identifier
            cleaned_response = cleaned_response.split("\n", 1)[1]  # Skip first line with ```[language]
            cleaned_response = cleaned_response.rsplit("```", 1)[0]  # Remove ending ``` and anything after it
        cleaned_response = cleaned_response.strip()
        
        data = json.loads(cleaned_response)
        if "rating" in data and "explanation" in data and data["rating"] in [0, 1, 2, 3, 4]:
            return cleaned_response
        return False
    except:
        return False

def rate_extract(evaluator_output: str) -> int:
    # First try JSON parsing
    try: 
        data = json.loads(evaluator_output)  # Parse JSON string
        rating = int(data["rating"])
        if rating not in [0, 1, 2, 3, 4]:
            return -1
        return rating
    except:
        # Fallback: try to find first number after "rating"
        try:
            import re
            match = re.search(r'rating["\s:\[]*(\d+)', evaluator_output)
            if match:
                rating = int(match.group(1))
                if rating in [0, 1, 2, 3, 4]:
                    return rating
        except:
            pass
        return -1
 

def mc_post_check(response_text):
    match = re.search(r"<choice>\s*(AnswerA|AnswerB)\s*</choice>", response_text)
    if match:
        return response_text
    else:
        return False

def mc_extract(response_text):
    match = re.search(r"<choice>\s*(AnswerA|AnswerB)\s*</choice>", response_text)
    if match:
        return match.group(1)
    return -1
