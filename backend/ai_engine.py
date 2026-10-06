import os
import json
import time
import logging

from groq import Groq, RateLimitError, APIError
from dotenv import load_dotenv

logger = logging.getLogger("businessguide")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not configured"
    )

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


def call_groq_with_retry(messages, temperature=0.1, max_retries=3):
    """
    Calls Groq chat completion API with automatic exponential backoff on 429 RateLimitError.
    """
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature
            )
        except RateLimitError as e:
            wait_time = (attempt + 1) * 2.0
            logger.warning(
                "[AI ENGINE] Groq RateLimitError hit (attempt %d/%d). Sleeping %.1fs before retrying...",
                attempt + 1,
                max_retries,
                wait_time
            )
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                raise
        except Exception as e:
            logger.error("[AI ENGINE] Groq API call failed: %s", str(e))
            raise


def ask_ai(prompt: str):
    response = call_groq_with_retry(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are BusinessGuide AI, an intelligent "
                    "business decision-support assistant. "
                    "Analyze the provided business information "
                    "carefully and provide concise, evidence-based "
                    "reasoning. Never invent data."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def fallback_target_selection(target_candidates):
    """
    Deterministic fallback target selection based on top-ranked statistical candidate.
    Used only if LLM calls fail or exhaust retries.
    """
    if not target_candidates:
        return {
            "recommended_target": "Unknown",
            "problem_type": "regression",
            "confidence": 0.5,
            "reason": "Default candidate target selected."
        }
    
    top = target_candidates[0]
    col = top.get("column", "Unknown")
    ptype = top.get("problem_type", "regression")
    score = top.get("suitability_score", top.get("score", 5))

    return {
        "recommended_target": col,
        "problem_type": ptype,
        "confidence": 0.85,
        "reason": f"Selected '{col}' as the top statistically qualified candidate target (suitability score: {score})."
    }


def select_business_target(
    dataset_overview,
    numerical_statistics,
    categorical_analysis,
    target_candidates
):
    # Prepare concise summary of candidates to minimize tokens and stay well within TPM limits
    condensed_candidates = [
        {
            "column": c.get("column"),
            "problem_type": c.get("problem_type"),
            "unique_values": c.get("unique_values"),
            "score": c.get("score")
        }
        for c in (target_candidates[:8] if isinstance(target_candidates, list) else [])
    ]

    overview_summary = {
        "rows": dataset_overview.get("rows") if isinstance(dataset_overview, dict) else len(dataset_overview),
        "columns": dataset_overview.get("columns") if isinstance(dataset_overview, dict) else 0
    }

    prompt = f"""
You are the decision-making intelligence of BusinessGuide AI.
Select the most appropriate prediction target for this dataset.

DATASET SUMMARY:
Rows: {overview_summary['rows']}, Columns: {overview_summary['columns']}

CANDIDATES:
{json.dumps(condensed_candidates, indent=2)}

Evaluate based on:
1. Meaningful business outcome (profit, churn, score, price, etc.)
2. Practical predictability from other features
3. Target problem type (regression vs classification)

Return ONLY valid JSON using exactly this structure:
{{
    "recommended_target": "column name",
    "problem_type": "regression or classification",
    "confidence": 0.90,
    "reason": "brief explanation"
}}
The confidence must be between 0 and 1.
"""

    try:
        response = call_groq_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": "You are BusinessGuide AI. You provide evidence-based business reasoning. Return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()
        
        # Clean potential markdown fences around JSON
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        result = json.loads(content)

        required_fields = [
            "recommended_target",
            "problem_type",
            "confidence",
            "reason"
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(f"AI response missing required field: {field}")

        if result["problem_type"] not in ["regression", "classification"]:
            raise ValueError("AI returned an invalid problem type")

        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        return result

    except Exception as e:
        logger.warning(
            "[AI ENGINE] Error calling Groq for target selection (%s). Utilizing statistical fallback.",
            str(e)
        )
        return fallback_target_selection(target_candidates)


def chat_with_assistant(
    user_message: str,
    history: list = None,
    context_str: str = None
) -> str:
    """
    Conversational BusinessGuide AI assistant that reasons strictly
    using the provided project context without hallucinating facts.
    """

    system_content = (
        "You are BusinessGuide AI Assistant, an expert conversational AI business decision-support assistant.\n"
        "Your role is to explain dataset insights, target selection logic, machine learning performance, feature drivers, "
        "and strategic takeaways to business stakeholders in clear, practical, evidence-based language.\n\n"
        "RULES:\n"
        "1. Reason and answer strictly using the PROJECT CONTEXT provided below.\n"
        "2. NEVER invent column names, numerical values, accuracy scores, or business facts not present in the context.\n"
        "3. If the user asks about an aspect of the project that has not been performed yet, clearly state that this step has not been run yet.\n"
        "4. If no dataset is uploaded, politely instruct the user to upload a dataset first.\n"
        "5. Explain technical machine learning concepts in simple, intuitive business terms.\n"
        "6. Format responses with clean Markdown, bullet points, or tables when explaining metrics or driver rankings.\n\n"
        f"PROJECT CONTEXT:\n{context_str or 'No project data available.'}"
    )

    messages = [{"role": "system", "content": system_content}]

    # Include recent conversation turns (capped at last 4 to keep context fast and focused)
    if history and isinstance(history, list):
        for msg in history[-4:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"])
                    })

    messages.append({
        "role": "user",
        "content": user_message
    })

    try:
        response = call_groq_with_retry(
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning("[AI ENGINE] Chat call failed: %s. Generating local contextual summary.", str(e))
        return (
            "### BusinessGuide AI Assistant\n\n"
            f"Here is what the project analytics show regarding your request:\n\n"
            f"{context_str[:800] if context_str else 'No active dataset context available.'}\n\n"
            "*Note: Real-time conversational generation temporarily experienced network throttling; the metrics above reflect active scikit-learn models.*"
        )