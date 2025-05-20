import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import PromptTemplate

from models.models import (
    LLM,
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


class GraphGeneration:
    def __init__(self, model: LLM):
        self.model = model
        prompt = load_prompt("sa-sgg.txt")
        self.prompt_template = PromptTemplate(
            input_variables=[
                "context",
                "target_sentence",
                "available_nodes",
                "available_edges",
            ],
            template=prompt,
        )

    def invoke(self, context, target_sentence, available_nodes, available_edges, verbs):
        all_nodes = f"{available_nodes}, {', '.join(verbs)}"

        prompt = self.prompt_template.format(
            context=context,
            target_sentence=target_sentence,
            available_nodes=all_nodes,
            available_edges=available_edges,
        )
        response = self.model.invoke(prompt)

        return response


class GraphScorer:
    @staticmethod
    def parse_response(response: str) -> List[List[List[str]]]:
        scene_graphs = []
        current_graph = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:  # If an empty line is encountered, start a new scene graph
                if current_graph:
                    scene_graphs.append(current_graph)
                    current_graph = []
                continue

            parts = [p.strip() for p in line.split("->")]
            if len(parts) == 3:  # Add only if it is a valid triplet
                current_graph.append(parts)

        if current_graph:  # Add the last graph
            scene_graphs.append(current_graph)

        return scene_graphs

    @staticmethod
    def calculate_scores(
        true_triplets: List[List[str]], pred_triplets: List[List[str]]
    ) -> Dict[str, float]:
        true_triplet_strs = set([" ".join(t) for t in true_triplets])
        pred_triplet_strs = set([" ".join(t) for t in pred_triplets])

        correct = len(true_triplet_strs.intersection(pred_triplet_strs))

        precision = correct / len(pred_triplet_strs) if pred_triplet_strs else 0
        recall = correct / len(true_triplet_strs) if true_triplet_strs else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {"precision": precision, "recall": recall, "f1": f1}


class GraphEvaluator:
    def __init__(self, model):
        self.generation = GraphGeneration(model)
        self.model_name = model.__class__.__name__

    def process_single_data(self, data: Dict) -> Dict:
        context = data["context"]
        target_sentence = data["target_sentence"]
        available_nodes = ", ".join(data["mandatory_space"]["object"])
        available_edges = ", ".join(data["mandatory_space"]["relationship"])
        verbs = data["mandatory_space"]["verb"]

        response = self.generation.invoke(
            context=context,
            target_sentence=target_sentence,
            available_nodes=available_nodes,
            available_edges=available_edges,
            verbs=verbs,
        )

        pred_scene_graphs = GraphScorer.parse_response(response)
        results = []

        # Match each predicted scene graph with the actual graph for evaluation
        for i, (pred_graph, true_graph) in enumerate(
            zip(pred_scene_graphs, data["graphs"])
        ):
            scores = GraphScorer.calculate_scores(true_graph["triplets"], pred_graph)

            true_triplet_set = set(tuple(t) for t in true_graph["triplets"])
            pred_triplet_set = set(tuple(t) for t in pred_graph)

            missing_triplets = true_triplet_set - pred_triplet_set
            incorrect_triplets = pred_triplet_set - true_triplet_set

            result = {
                "input_data": data,
                "model_response": response,
                "graph_id": true_graph.get("action_id"),
                "predicted_triplets": pred_graph,
                "true_triplets": true_graph["triplets"],
                "missing_triplets": list(missing_triplets),
                "incorrect_triplets": list(incorrect_triplets),
                **scores,
            }
            results.append(result)

        return results

    def evaluate(self, data_path: str):
        processed_data = []
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_graphs = 0

        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()  # 모든 라인을 미리 읽어옴

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {
                executor.submit(self.process_single_data, json.loads(line)): line
                for line in lines
            }
            for future in as_completed(futures):
                results = future.result()
                processed_data.extend(results)

                for result in results:
                    total_precision += result["precision"]
                    total_recall += result["recall"]
                    total_f1 += result["f1"]
                    total_graphs += 1

        macro_metrics = {
            "model": self.model_name,
            "macro_precision": total_precision / total_graphs,
            "macro_recall": total_recall / total_graphs,
            "macro_f1": total_f1 / total_graphs,
        }

        return macro_metrics


def evaluate_graph_generation(model: LLM):
    evaluator = GraphEvaluator(model)
    return evaluator.evaluate(
        "resource/dataset/generation/sa-sgg.jsonl"
    )


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
        results = evaluate_graph_generation(model)
        print(f"Evaluation Results for {model.__class__.__name__}:", results)
