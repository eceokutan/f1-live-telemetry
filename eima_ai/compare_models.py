#!/usr/bin/env python3
"""
Model Comparison Script: Base Granite vs Fine-Tuned Granite

Compares responses from the base IBM Granite model against a fine-tuned version
using F1 race-specific questions. Results are displayed side-by-side and can be
saved to a file for analysis.

Usage:
    python compare_models.py                    # Run all questions
    python compare_models.py --questions 3      # Run first 3 questions
    python compare_models.py --save             # Save results to JSON
    python compare_models.py --custom "question" # Ask a custom question
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_ibm import WatsonxLLM


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base model (production IBM Granite)
BASE_MODEL_ID = "ibm/granite-3-8b-instruct"

# Fine-tuned model ID - UPDATE THIS when training is complete
# This should be the deployment ID or model ID of your fine-tuned model
FINETUNED_MODEL_ID = "YOUR_FINETUNED_MODEL_ID_HERE"  # TODO: Update this

# Model parameters
MODEL_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0.7,
    "decoding_method": "greedy",
}


# =============================================================================
# F1 RACE-SPECIFIC TEST QUESTIONS
# =============================================================================

RACE_QUESTIONS = [
    # Tire management questions
    {
        "category": "Tire Management",
        "question": "My front left tire is at 98°C and degrading faster than the others. What should I do?",
        "context": "Lap 23 of 56, medium compound tires, 15 laps old"
    },
    {
        "category": "Tire Management",
        "question": "Should I push harder on my outlap to get heat into the tires or take it easy?",
        "context": "Just exited pit lane on fresh soft tires, track temp 35°C"
    },
    {
        "category": "Tire Management",
        "question": "I'm experiencing graining on the front tires. How should I adjust my driving?",
        "context": "Lap 8 of 52, hard compound, tire temps 85°C front, 90°C rear"
    },

    # Fuel management questions
    {
        "category": "Fuel Management",
        "question": "I have 12 laps of fuel remaining but 15 laps to go. What's the best strategy?",
        "context": "P4, 3.2s gap to P3, 4.1s gap to P5 behind"
    },
    {
        "category": "Fuel Management",
        "question": "Can I do one more push lap before fuel saving becomes critical?",
        "context": "Qualifying simulation, 2.3 laps of fuel remaining"
    },

    # Race strategy questions
    {
        "category": "Race Strategy",
        "question": "The car ahead just pitted. Should I stay out or box this lap?",
        "context": "Lap 34 of 58, P3, gap to leader 8.2s, pit window open for 5 more laps"
    },
    {
        "category": "Race Strategy",
        "question": "Safety car is out. What are our options?",
        "context": "Lap 41 of 66, P5, on 25-lap old mediums, 18s to pit lane"
    },
    {
        "category": "Race Strategy",
        "question": "DRS train ahead of me. Should I save tires and wait or push to break through?",
        "context": "P7, cars in P4-P6 within 1.5s of each other, 12 laps remaining"
    },

    # Car setup and handling questions
    {
        "category": "Car Handling",
        "question": "I'm getting massive understeer in the slow corners. Any advice?",
        "context": "Sector 3 times dropping, front tire temps 92°C, rear 88°C"
    },
    {
        "category": "Car Handling",
        "question": "The rear is stepping out on corner exit. What can I do?",
        "context": "High-speed corners, rear tire wear at 45%, traction control at level 3"
    },

    # Gap and position questions
    {
        "category": "Race Position",
        "question": "How much do I need to push to catch the car ahead before DRS range?",
        "context": "Gap to P3: 1.8s, current pace delta: +0.3s per lap"
    },
    {
        "category": "Race Position",
        "question": "Car behind is closing at 0.5s per lap. How many laps until they catch me?",
        "context": "Current gap: 4.2s, 20 laps remaining"
    },

    # Weather and conditions
    {
        "category": "Weather",
        "question": "Rain is forecast in 10 minutes. Should we pit early for inters?",
        "context": "P6, on softs, competitors ahead all on mediums"
    },
    {
        "category": "Track Conditions",
        "question": "Track is rubbering in. Should I adjust my line through turn 4?",
        "context": "Lap 35, mid-race, traditional line becoming slippery"
    },

    # Technical questions
    {
        "category": "Technical",
        "question": "Battery deployment seems low in sector 2. Is something wrong?",
        "context": "Hybrid system showing 78% deployment vs target 85%"
    },
    {
        "category": "Technical",
        "question": "Brake temps are climbing. Should I adjust brake bias?",
        "context": "Front brake temps: 850°C, optimal range 700-800°C"
    },
]


# =============================================================================
# MODEL CLIENT CLASS
# =============================================================================

class GraniteModelClient:
    """Client for interacting with IBM Granite models."""

    def __init__(
        self,
        model_id: str,
        watsonx_url: str,
        watsonx_project_id: str,
        watsonx_api_key: str,
        name: str = "Model"
    ):
        self.model_id = model_id
        self.name = name
        self.llm = WatsonxLLM(
            model_id=model_id,
            url=watsonx_url,
            project_id=watsonx_project_id,
            apikey=watsonx_api_key,
            params=MODEL_PARAMS,
        )

    async def ask(self, question: str, context: str = "") -> str:
        """Ask the model a question and return the response."""
        prompt = self._format_prompt(question, context)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.llm.invoke, prompt)
            return response.strip()
        except Exception as e:
            return f"[ERROR: {str(e)}]"

    def _format_prompt(self, question: str, context: str) -> str:
        """Format the question with F1 race engineer context."""
        system_context = """You are an expert F1 race engineer communicating with your driver during a race.
Provide concise, actionable advice. The driver is actively racing and cannot read long responses.
Be direct, use precise numbers when helpful, and prioritize the most critical information first."""

        if context:
            return f"""{system_context}

Current Situation: {context}

Driver: {question}

Race Engineer:"""
        else:
            return f"""{system_context}

Driver: {question}

Race Engineer:"""


# =============================================================================
# COMPARISON ENGINE
# =============================================================================

class ModelComparator:
    """Compares responses between base and fine-tuned models."""

    def __init__(self):
        self.watsonx_url = os.getenv("WATSONX_URL")
        self.watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
        self.watsonx_api_key = os.getenv("WATSONX_API_KEY")

        self._validate_credentials()

        self.base_model: Optional[GraniteModelClient] = None
        self.finetuned_model: Optional[GraniteModelClient] = None
        self.results = []

    def _validate_credentials(self):
        """Validate that all required credentials are present."""
        missing = []
        if not self.watsonx_url:
            missing.append("WATSONX_URL")
        if not self.watsonx_project_id:
            missing.append("WATSONX_PROJECT_ID")
        if not self.watsonx_api_key:
            missing.append("WATSONX_API_KEY")

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def initialize_models(self, finetuned_model_id: str):
        """Initialize both models for comparison."""
        print("\n" + "=" * 70)
        print("INITIALIZING MODELS")
        print("=" * 70)

        # Initialize base model
        print(f"\n[1/2] Initializing base model: {BASE_MODEL_ID}")
        self.base_model = GraniteModelClient(
            model_id=BASE_MODEL_ID,
            watsonx_url=self.watsonx_url,
            watsonx_project_id=self.watsonx_project_id,
            watsonx_api_key=self.watsonx_api_key,
            name="Base Granite"
        )
        print("      Base model ready.")

        # Initialize fine-tuned model
        print(f"\n[2/2] Initializing fine-tuned model: {finetuned_model_id}")
        if finetuned_model_id == "YOUR_FINETUNED_MODEL_ID_HERE":
            print("      WARNING: Using placeholder model ID!")
            print("      Update FINETUNED_MODEL_ID in compare_models.py when training is complete.")
            print("      For now, using base model as stand-in for comparison demo.")
            finetuned_model_id = BASE_MODEL_ID

        self.finetuned_model = GraniteModelClient(
            model_id=finetuned_model_id,
            watsonx_url=self.watsonx_url,
            watsonx_project_id=self.watsonx_project_id,
            watsonx_api_key=self.watsonx_api_key,
            name="Fine-tuned Granite"
        )
        print("      Fine-tuned model ready.")
        print("\n" + "=" * 70)

    async def compare_question(self, question: str, context: str = "", category: str = "General") -> dict:
        """Ask both models the same question and compare responses."""
        print(f"\n{'─' * 70}")
        print(f"CATEGORY: {category}")
        print(f"{'─' * 70}")
        print(f"\nQUESTION: {question}")
        if context:
            print(f"CONTEXT:  {context}")
        print()

        # Query both models in parallel
        base_task = self.base_model.ask(question, context)
        finetuned_task = self.finetuned_model.ask(question, context)

        base_response, finetuned_response = await asyncio.gather(base_task, finetuned_task)

        # Display results side by side
        print("┌" + "─" * 33 + "┬" + "─" * 33 + "┐")
        print("│" + " BASE MODEL".center(33) + "│" + " FINE-TUNED MODEL".center(33) + "│")
        print("├" + "─" * 33 + "┼" + "─" * 33 + "┤")

        # Wrap text for display
        base_lines = self._wrap_text(base_response, 31)
        finetuned_lines = self._wrap_text(finetuned_response, 31)
        max_lines = max(len(base_lines), len(finetuned_lines))

        for i in range(max_lines):
            base_line = base_lines[i] if i < len(base_lines) else ""
            finetuned_line = finetuned_lines[i] if i < len(finetuned_lines) else ""
            print(f"│ {base_line:<31} │ {finetuned_line:<31} │")

        print("└" + "─" * 33 + "┴" + "─" * 33 + "┘")

        result = {
            "category": category,
            "question": question,
            "context": context,
            "base_response": base_response,
            "finetuned_response": finetuned_response,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        return result

    def _wrap_text(self, text: str, width: int) -> list:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                # Handle words longer than width
                while len(word) > width:
                    lines.append(word[:width])
                    word = word[width:]
                current_line = [word] if word else []
                current_length = len(word) + 1 if word else 0

        if current_line:
            lines.append(" ".join(current_line))

        return lines if lines else [""]

    async def run_all_comparisons(self, num_questions: Optional[int] = None):
        """Run comparisons for all (or specified number of) test questions."""
        questions = RACE_QUESTIONS[:num_questions] if num_questions else RACE_QUESTIONS

        print("\n" + "=" * 70)
        print(f"RUNNING {len(questions)} COMPARISON(S)")
        print("=" * 70)

        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]", end="")
            await self.compare_question(
                question=q["question"],
                context=q.get("context", ""),
                category=q["category"]
            )

            # Small delay between questions to avoid rate limiting
            if i < len(questions):
                await asyncio.sleep(1)

    async def run_custom_question(self, question: str, context: str = ""):
        """Run comparison for a custom question."""
        await self.compare_question(question, context, "Custom Question")

    def save_results(self, filename: str = None):
        """Save comparison results to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"

        output = {
            "metadata": {
                "base_model": BASE_MODEL_ID,
                "finetuned_model": FINETUNED_MODEL_ID,
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(self.results)
            },
            "results": self.results
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {filename}")
        return filename

    def print_summary(self):
        """Print a summary of the comparison results."""
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nTotal questions tested: {len(self.results)}")

        # Group by category
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1

        print("\nQuestions by category:")
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count}")

        print("\n" + "=" * 70)
        print("NOTE: Review the responses above to evaluate model quality.")
        print("Consider factors like: accuracy, conciseness, F1 terminology,")
        print("and actionability of advice.")
        print("=" * 70)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned IBM Granite models on F1 questions"
    )
    parser.add_argument(
        "--questions", "-n",
        type=int,
        help="Number of test questions to run (default: all)"
    )
    parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save results to a JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output filename for results (requires --save)"
    )
    parser.add_argument(
        "--custom", "-c",
        type=str,
        help="Ask a custom question instead of predefined ones"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="",
        help="Context for custom question (optional)"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default=FINETUNED_MODEL_ID,
        help=f"Fine-tuned model ID (default: {FINETUNED_MODEL_ID})"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("F1 RACE ENGINEER MODEL COMPARISON TOOL")
    print("=" * 70)
    print(f"\nBase Model:       {BASE_MODEL_ID}")
    print(f"Fine-tuned Model: {args.finetuned_model}")

    try:
        comparator = ModelComparator()
        comparator.initialize_models(args.finetuned_model)

        if args.custom:
            await comparator.run_custom_question(args.custom, args.context)
        else:
            await comparator.run_all_comparisons(args.questions)

        comparator.print_summary()

        if args.save:
            comparator.save_results(args.output)

    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure your .env file contains:")
        print("  WATSONX_URL=https://...")
        print("  WATSONX_PROJECT_ID=...")
        print("  WATSONX_API_KEY=...")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
