import concurrent.futures
import json
import re
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate

from models.models import (
    GPT4o,
    GPT4oMini,
    Claude35Sonnet,
    Claude35Haiku,
    MetaLlama,
    Qwen,
    DeepSeek,
    MistralMixtral,
    MistralLarge,
    Qwen7B,
    Mistral7B,
)
from utils.path import load_prompt


class SceneGraphToText:
    def __init__(self, model: GPT4o):
        self.model = model
        prompt = load_prompt("sgds.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["sentences", "triplet", "context"],
            template=prompt,
        )

    def invoke(self, sentences, triplet, context_graphs):
        prompt = self.prompt_template.format(
            sentences=sentences, triplet=triplet, context=context_graphs
        )
        response = self.model.invoke(prompt)
        return response


class SceneGraphEvaluator:
    def __init__(self, model: GPT4o):
        self.scene_graph_to_text = SceneGraphToText(model)
        self.model_name = model.__class__.__name__

    def parse_prediction(self, response: str) -> int:
        pattern = r"\[([A-E])\]|\b([A-E])\b"
        match = re.search(pattern, response)
        if match:
            # Use the first group (inside brackets) or the second group (standalone letter)
            letter = match.group(1) or match.group(2)
            # Convert 'A' to 0, 'B' to 1, etc.
            return ord(letter) - 65
        return None  # Return None if no alphabet is found

    def process_single_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item."""
        result = data.copy()

        variations_str = "\n".join(
            [
                f"{chr(65 + i)}: {variation}"  # 65 is the ASCII code for 'A'
                for i, variation in enumerate(data["variations"])
            ]
        )

        response = self.scene_graph_to_text.invoke(
            variations_str, data["triplet"], data["context_graphs"]
        )

        # 알파벳 응답을 숫자로 변환
        prediction = self.parse_prediction(response)
        if prediction is not None:
            result["prediction"] = prediction
            result["is_correct"] = prediction == data["position"]
        else:
            result["prediction"] = None
            result["is_correct"] = False

        return result

    def evaluate_dataset(self, input_path: str):
        """Evaluate the entire dataset and return metrics."""
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        correct_count = 0
        total_count = len(lines)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {
                executor.submit(
                    self.process_single_data, json.loads(line.strip())
                ): line
                for line in lines
            }

            for future in concurrent.futures.as_completed(futures):
                processed_data = future.result()

                if processed_data["is_correct"]:
                    correct_count += 1

        # Final metric calculation
        accuracy = correct_count / total_count if total_count > 0 else 0
        metrics = {
            "model": self.model_name,
            "accuracy": accuracy,
            "total_samples": total_count,
            "correct_predictions": correct_count,
        }
        return metrics


def evaluate_graph_to_text(model: GPT4o, input_path: str):
    evaluator = SceneGraphEvaluator(model)
    return evaluator.evaluate_dataset(input_path)


def main():
    model_classes = [
        GPT4o,
        GPT4oMini,
        Claude35Sonnet,
        Claude35Haiku,
        MetaLlama,
        Qwen,
        DeepSeek,
        MistralMixtral,
        MistralLarge,
        Qwen7B,
        Mistral7B,
    ]
    input_path = "resource/dataset/understaing/sgds.jsonl"

    for model_class in model_classes:
        model = model_class()
        results = evaluate_graph_to_text(model, input_path)
        print(f"Evaluation result ({model.__class__.__name__}):", results)


if __name__ == "__main__":
    main()
