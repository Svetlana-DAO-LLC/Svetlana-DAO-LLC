"""
CircAIdian Memory Benchmark Harness — LoCoMo-style evaluation.

Metrics:
1. Memory Accuracy — Does the agent correctly answer questions about disclosed facts?
2. Adversarial Robustness — Does the agent refuse questions about undisclosed facts?
3. Token Efficiency — How many tokens does CircAIdian use vs full-context replay?

Usage:
    python -m benchmark.harness
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon import ActiveContextManager, EmotionalState, CorrectionHandler, NudgeEngine
from benchmark.locomo_simulated import SCENARIOS, ConversationScenario
from benchmark.bm25 import bm25_retrieve


class MemoryBenchmarkHarness:
    """Evaluates CircAIdian's memory system against LoCoMo-style scenarios."""

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens
        self.results: List[Dict] = []

    async def load_scenario(self, scenario: ConversationScenario) -> ActiveContextManager:
        """Load a scenario's conversation turns into the context manager."""
        import tempfile
        from pathlib import Path

        # Need a real temp DB path (can't use /dev/null)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "corrections.db"

            cm = ActiveContextManager(max_context_tokens=self.max_context_tokens)
            es = EmotionalState()
            ch = CorrectionHandler(soul_path="/dev/null", db_path=str(db_path))
            ne = NudgeEngine()

            for user_msg, agent_response in scenario.turns:
                await ch.process_observation(
                    session_id=scenario.scenario_id,
                    user_msg=user_msg,
                    agent_output=agent_response,
                    soul_content="",
                )
                cm.add_message_pair(user_msg, agent_response)

            return cm

    def _retrieve_context(self, cm: ActiveContextManager, query: str) -> str:
        """Retrieve relevant context using BM25 ranking."""
        chunks = cm.get_all_chunks()
        if not chunks:
            return ""

        chunk_tuples = [(c.chunk_id, c.content) for c in chunks]
        return bm25_retrieve(query, chunk_tuples, top_k=5)

    async def evaluate_scenario(self, scenario: ConversationScenario) -> Dict:
        """Run Q&A evaluation for one scenario."""
        cm = await self.load_scenario(scenario)

        # Full context size (for token efficiency comparison)
        full_context = cm.get_context_for_prompt()
        full_tokens = cm.estimated_tokens

        # CircAIdian retrieval context
        retrieved_contexts = []
        for qa in scenario.qa_pairs:
            ctx = self._retrieve_context(cm, qa.question)
            retrieved_contexts.append(ctx)

        retrieved_tokens = sum(len(ctx.split()) for ctx in retrieved_contexts)

        # Simulate LLM answering using retrieved context
        # For benchmark purposes: check if relevant keywords appear in retrieved context
        disclosed_correct = 0
        disclosed_total = 0
        adversarial_correct_refusal = 0
        adversarial_total = 0

        details = []

        for qa, retrieved_ctx in zip(scenario.qa_pairs, retrieved_contexts):
            if qa.is_adversarial:
                # Adversarial: should NOT find the answer
                adversarial_total += 1
                # Check that the answer keywords are NOT in retrieved context
                answer_keywords = {
                    w.lower() for w in qa.correct_answer.split()
                    if w.lower() not in ("undisclosed", "not")
                }
                found_in_context = any(
                    kw in retrieved_ctx.lower()
                    for kw in answer_keywords
                    if len(kw) > 3
                )
                if not found_in_context:
                    adversarial_correct_refusal += 1
                    result = "CORRECT_REFUSAL"
                else:
                    result = "HALLUCINATED"  # Answered something never said
                details.append({"q": qa.question, "expected": qa.correct_answer, "result": result})
            else:
                # Disclosed: should find the answer
                disclosed_total += 1
                answer_keywords = {
                    w.lower() for w in qa.correct_answer.split()
                    if len(w) > 2
                }
                found = sum(
                    1 for kw in answer_keywords
                    if kw in retrieved_ctx.lower()
                )
                # Answerable if most keywords found (>50%)
                if found >= len(answer_keywords) * 0.5:
                    disclosed_correct += 1
                    result = "CORRECT"
                else:
                    result = "INCORRECT"
                details.append({
                    "q": qa.question,
                    "expected": qa.correct_answer,
                    "retrieved_snippet": retrieved_ctx[:100],
                    "result": result,
                })

        memory_accuracy = disclosed_correct / disclosed_total if disclosed_total else 0
        adversarial_robustness = (
            adversarial_correct_refusal / adversarial_total if adversarial_total else 0
        )
        token_efficiency = 1 - (retrieved_tokens / (full_tokens * len(scenario.qa_pairs))) if full_tokens else 0

        return {
            "scenario_id": scenario.scenario_id,
            "persona": scenario.persona,
            "memory_accuracy": memory_accuracy,
            "adversarial_robustness": adversarial_robustness,
            "token_efficiency": max(0, token_efficiency),
            "disclosed_correct": disclosed_correct,
            "disclosed_total": disclosed_total,
            "adversarial_refused": adversarial_correct_refusal,
            "adversarial_total": adversarial_total,
            "full_context_tokens": full_tokens,
            "avg_retrieved_tokens": retrieved_tokens / len(scenario.qa_pairs) if scenario.qa_pairs else 0,
            "details": details,
        }

    async def run(self) -> Dict:
        """Run full benchmark across all scenarios."""
        print("=" * 60)
        print("CircAIdian Memory Benchmark — LoCoMo-style")
        print("=" * 60)
        print(f"Scenarios: {len(SCENARIOS)}")
        print(f"Max context tokens: {self.max_context_tokens}")
        print()

        all_results = []
        for scenario in SCENARIOS:
            print(f"Evaluating: {scenario.scenario_id} ({scenario.persona[:40]}...)")
            result = await self.evaluate_scenario(scenario)
            all_results.append(result)
            print(f"  Memory Accuracy:    {result['memory_accuracy']:.1%}")
            print(f"  Adversarial Robust: {result['adversarial_robustness']:.1%}")
            print(f"  Token Efficiency:   {result['token_efficiency']:.1%}")
            print()

        # Aggregate
        n = len(all_results)
        avg_memory_acc = sum(r["memory_accuracy"] for r in all_results) / n
        avg_adversarial = sum(r["adversarial_robustness"] for r in all_results) / n
        avg_token_eff = sum(r["token_efficiency"] for r in all_results) / n
        total_disclosed = sum(r["disclosed_total"] for r in all_results)
        total_adversarial = sum(r["adversarial_total"] for r in all_results)
        total_disclosed_correct = sum(r["disclosed_correct"] for r in all_results)
        total_adversarial_refused = sum(r["adversarial_refused"] for r in all_results)

        print("=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(f"Overall Memory Accuracy:      {avg_memory_acc:.1%}  ({total_disclosed_correct}/{total_disclosed})")
        print(f"Overall Adversarial Robust:   {avg_adversarial:.1%}  ({total_adversarial_refused}/{total_adversarial})")
        print(f"Overall Token Efficiency:      {avg_token_eff:.1%}")
        print()

        # Per-scenario breakdown
        print(f"{'Scenario':<25} {'Mem Acc':>8} {'Adv Rob':>8} {'Token Eff':>10}")
        print("-" * 55)
        for r in all_results:
            print(f"{r['scenario_id']:<25} {r['memory_accuracy']:>7.1%} {r['adversarial_robustness']:>7.1%} {r['token_efficiency']:>9.1%}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "max_context_tokens": self.max_context_tokens,
            "num_scenarios": n,
            "overall_memory_accuracy": avg_memory_acc,
            "overall_adversarial_robustness": avg_adversarial,
            "overall_token_efficiency": avg_token_eff,
            "total_disclosed_correct": total_disclosed_correct,
            "total_disclosed": total_disclosed,
            "total_adversarial_refused": total_adversarial_refused,
            "total_adversarial": total_adversarial,
            "per_scenario": [
                {
                    "scenario_id": r["scenario_id"],
                    "memory_accuracy": r["memory_accuracy"],
                    "adversarial_robustness": r["adversarial_robustness"],
                    "token_efficiency": r["token_efficiency"],
                }
                for r in all_results
            ],
        }

        return summary


async def main():
    harness = MemoryBenchmarkHarness(max_context_tokens=4000)
    summary = await harness.run()

    # Save results
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
