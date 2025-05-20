import concurrent.futures
import json
import re
from pathlib import Path
from typing import Dict, List

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
from utils.path import get_project_path, load_prompt


class QA:
    def __init__(self, model: GPT4o):
        self.model = model
        prompt = load_prompt("sgqa.txt")
        self.prompt_template = PromptTemplate(
            input_variables=[
                "scene_graph",
                "question",
            ],
            template=prompt,
        )

    def invoke(self, scene_graph, question):
        prompt = self.prompt_template.format(
            scene_graph=scene_graph,
            question=question,
        )
        response = self.model.invoke(prompt)
        answer = re.findall(r"\[(.*?)\]", response)
        return answer[0] if answer else response


class QADataLoader:
    def __init__(self):
        self.qa_path = (
            Path(get_project_path())
            / "resource"
            / "dataset"
            / "understaing"
            / "sgqa.jsonl"
        )
        self.qa_data = self._load_jsonl(self.qa_path)

    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data

    def get_data(self):
        data = []
        for qa_item in self.qa_data:
            for qa_pair in qa_item["qa_pairs"]:
                data.append(
                    {
                        "data_id": qa_item["data_id"],
                        "doc_index": qa_item["doc_index"],
                        "text_part_index": qa_item["text_part_index"],
                        "context_graphs": qa_item["context_graphs"],
                        "question": qa_pair["Q"],
                        "answer": qa_pair["A"],
                    }
                )
        return data


class QAEvaluator:
    def __init__(self, model: GPT4o):
        self.inference = QA(model)
        self.model_name = model.__class__.__name__

    def process_single_question(self, data: Dict) -> Dict:
        prediction = self.inference.invoke(
            scene_graph=data["context_graphs"], question=data["question"]
        )

        is_correct = prediction.lower().strip() == data["answer"].lower().strip()

        result = {
            "data_id": data["data_id"],
            "doc_index": data["doc_index"],
            "text_part_index": data["text_part_index"],
            "context_graphs": data["context_graphs"],
            "question": data["question"],
            "true_answer": data["answer"],
            "predicted_answer": prediction,
            "is_correct": is_correct,
        }

        return result

    def evaluate(self, data_loader: QADataLoader):
        processed_data = []
        total_correct = 0
        total_questions = 0

        total_samples = len(data_loader.get_data())
        print(f"\nProcessing with {self.model_name} ({total_samples} samples)")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, data in enumerate(data_loader.get_data(), 1):
                futures.append(
                    (executor.submit(self.process_single_question, data), idx)
                )

            for future, idx in futures:
                processed = future.result()
                processed_data.append(processed)

                if processed["is_correct"]:
                    total_correct += 1
                total_questions += 1

                print(
                    f"\r[{self.model_name}] {idx}/{total_samples} samples processed",
                    end="",
                )

        print()  # 새 줄로 이동

        accuracy = total_correct / total_questions if total_questions > 0 else 0

        metrics = {
            "model": self.model_name,
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
        }

        return metrics


def evaluate_qa(model: GPT4o):
    data_loader = QADataLoader()
    evaluator = QAEvaluator(model)
    return evaluator.evaluate(data_loader)


if __name__ == "__main__":
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

    for model_class in model_classes:
        model = model_class()
        results = evaluate_qa(model)
        print(f"Evaluation Results for {model.__class__.__name__}:", results)
