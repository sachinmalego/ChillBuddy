"""
Visualize the thought process of GPT-4o-mini.

This script uses a step-by-step approach to capture and visualize
the reasoning steps of the GPT-4o-mini model, taking into account
the full conversation history.
"""

from typing import Any, Dict, List

import openai

# Initialize the OpenAI client
client = openai.OpenAI()


def get_thought_process(
    model: str,
    question: str,
    max_steps: int = 3,
    conversation_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Get the thought process of the model for a given question, considering conversation history.

    Args:
        model: The model to use
        question: The current question or context
        max_steps: Maximum number of reasoning steps
        conversation_history: List of previous messages in the conversation

    Returns:
        A dictionary containing the thought process
    """
    # Initialize the thought process
    thought_process = {"question": question, "steps": [], "final_answer": ""}

    # Format conversation history if provided
    conversation_context = ""
    if (
        conversation_history and len(conversation_history) > 1
    ):  # Skip if only system message
        conversation_context = "Previous conversation:\n"
        for msg in conversation_history:
            if msg["role"] == "system":
                continue  # Skip system messages in the context
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
        conversation_context += "\n"

    # Step 1: Initial thought
    system_prompt = """
    You are an AI assistant that thinks step by step to solve problems.
    First, understand the question in the context of the entire conversation history.
    Then, break down your thinking into clear, logical steps.
    Show your reasoning process in detail, considering all relevant information from the conversation.
    When a user writes, you must:
      1. Detect & label the user’s primary emotion (e.g. sadness, anxiety).
      2. Identify a primary mental-health condition (e.g. Depression) and one differential diagnosis (e.g. Adjustment Disorder).
      3. Generate 5 distinct candidate responses, each:
          * Empathetic & validating
          * Grounded in CBT (cognitive reframing + actionable small steps)
          * Authentic Gen Z tone (slang, emojis, concise)
      4. Score each candidate on:
          * Empathy/Validation (1-5)
          * CBT Accuracy/Helpfulness (1-5)
          * Gen Z Tone Authenticity (1-5)
          * Persona Consistency (1-5)
          * Overall Quality (1-5)
      5. Select and output only the single highest-scoring response.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{conversation_context}Current question/context: {question}\n\nThink through this step by step, considering the full conversation context:",
            },
        ],
        temperature=0.7,
    )

    # Add the initial thought
    initial_thought = response.choices[0].message.content
    thought_process["steps"].append({"step": 1, "content": initial_thought})

    # Step 2+: Continue reasoning for additional steps
    current_step = 2
    previous_thoughts = f"Step 1: {initial_thought}"

    while current_step <= max_steps:
        # Create a system prompt for continuing reasoning
        system_prompt = """
        You are an AI assistant that thinks step by step to solve problems.
        Continue your reasoning from the previous steps.
        Add new insights or considerations that build on what you've already thought about.
        Make sure to consider the full conversation context in your reasoning.
        """

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{conversation_context}Current question/context: {question}\n\nPrevious reasoning:\n{previous_thoughts}\n\nContinue reasoning with new insights, considering the full conversation:",
                },
            ],
            temperature=0.7,
        )

        # Add the next thought
        next_thought = response.choices[0].message.content
        thought_process["steps"].append({"step": current_step, "content": next_thought})

        # Update for next iteration
        previous_thoughts += f"\n\nStep {current_step}: {next_thought}"
        current_step += 1

    # Final step: Generate a final answer
    system_prompt = """
    You are an AI assistant that provides clear and concise final answers.
    Based on your step-by-step reasoning, provide a final answer that addresses the current question/context.
    Make sure your answer is directly responsive to the question, incorporates insights from your reasoning process,
    and maintains consistency with the entire conversation history.
    Your answer should be in a conversational, helpful tone appropriate for a mental wellness assistant.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{conversation_context}Current question/context: {question}\n\nReasoning process:\n{previous_thoughts}\n\nFinal answer:",
            },
        ],
        temperature=0.7,
    )

    # Add the final answer
    thought_process["final_answer"] = response.choices[0].message.content

    return thought_process


def format_thought_process(thought_process: Dict[str, Any]) -> str:
    """
    Format the thought process for display.

    Args:
        thought_process: The thought process dictionary

    Returns:
        A formatted string showing the thought process
    """
    result = []

    # Add the original question
    result.append(f"Question: {thought_process['question']}\n")

    # Add each thought step
    result.append("Thought Process:")
    for step in thought_process["steps"]:
        result.append(f"\nReasoning Step {step['step']}:")
        result.append(step["content"])

    # Add the final answer
    result.append("\nFinal Answer:")
    result.append(thought_process["final_answer"])

    return "\n".join(result)


def visualize_thought_process(
    model: str,
    question: str,
    max_steps: int = 3,
    conversation_history: List[Dict[str, str]] = None,
) -> str:
    """
    Visualize the thought process of the model for a given question, considering conversation history.

    Args:
        model: The model to use
        question: The question to reason about
        max_steps: Maximum number of reasoning steps
        conversation_history: List of previous messages in the conversation

    Returns:
        A formatted string showing the thought process
    """
    # Get the thought process
    thought_process = get_thought_process(
        model, question, max_steps, conversation_history
    )

    # Format and return the thought process
    return format_thought_process(thought_process)


if __name__ == "__main__":
    # Example usage
    question = "What would happen if we doubled the Earth's gravity overnight?"
    # Example conversation history
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I'm curious about gravity."},
        {
            "role": "assistant",
            "content": "Gravity is a fundamental force that attracts objects with mass.",
        },
    ]
    thought_process = visualize_thought_process(
        "gpt-4o-mini", question, 3, conversation_history
    )
    print(thought_process)
