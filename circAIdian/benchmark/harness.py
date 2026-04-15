"""
CircAIdian Memory Benchmark Harness — LoCoMo-style evaluation.

Supports four retrieval modes:
  --mode bm25    : Pure BM25 keyword retrieval (baseline)
  --mode hyde    : HYDE (M2.7 hypothetical answer) + BM25
  --mode custom  : BM25 → HYDE → Subconscious dream (M2.7) → M2.7 rerank
  --mode qmd     : QMD hybrid lex+vec+hyde retrieval (CPU-constrained)

Metrics:
1. Memory Accuracy — Does the agent correctly answer questions about disclosed facts?
2. Adversarial Robustness — Does the agent refuse questions about undisclosed facts?
3. Token Efficiency — How many tokens does CircAIdian use vs full-context replay?

Usage:
    python -m benchmark.harness --mode bm25
    python -m benchmark.harness --mode hyde
    python -m benchmark.harness --mode qmd
"""
import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from daemon import ActiveContextManager, EmotionalState, CorrectionHandler, NudgeEngine
from benchmark.locomo_simulated import SCENARIOS, ConversationScenario
from benchmark.bm25 import bm25_retrieve
from benchmark.hyde_retrieval import HYDEBM25Retrieval, hyde_retrieve_async
from benchmark.qmd_retrieval import QMDRetrieval
from benchmark.custom_retrieval import CircAIdianRetrieval, CircAIdianLightRetrieval


class MemoryBenchmarkHarness:
    """Evaluates CircAIdian's memory system against LoCoMo-style scenarios."""

    def __init__(self, max_context_tokens: int = 4000, mode: str = "hyde"):
        self.max_context_tokens = max_context_tokens
        self.mode = mode  # 'bm25' | 'hyde' | 'qmd'
        self.results: List[Dict] = []

    async def load_scenario(
        self, scenario: ConversationScenario
    ) -> Tuple[ActiveContextManager, Optional[HYDEBM25Retrieval], Optional[QMDRetrieval], Optional[CircAIdianLightRetrieval]]:
        """
        Load a scenario's conversation turns into the context manager.

        Returns (cm, hyde_retriever, qmd_retriever, custom_retriever).
        hyde_retriever is built (and indexed) for HYDE mode.
        qmd_retriever is built (and indexed) for QMD mode.
        custom_retriever is built for custom mode (M2.7 rerank).
        Caller must call close() on any non-None retriever.
        """
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

            chunks = cm.get_all_chunks()
            chunk_tuples = [(c.chunk_id, c.content) for c in chunks]

            hyde: Optional[HYDEBM25Retrieval] = None
            qmd: Optional[QMDRetrieval] = None
            custom: Optional[CircAIdianLightRetrieval] = None

            if self.mode in ("hyde", "qmd", "custom") and chunks:
                if self.mode == "hyde":
                    hyde = HYDEBM25Retrieval()
                    hyde.index(chunk_tuples)
                elif self.mode == "qmd":
                    # Use the persistent locomo-bench collection (not per-scenario).
                    # It must exist and contain the scenario's chunk files.
                    # We skip index_chunks (which cleans up) and instead
                    # just verify the collection is registered and query it.
                    qmd = QMDRetrieval(collection_name="locomo-bench", persistent=True)
                    if qmd.query("__ping__", top_k=1):
                        qmd._indexed = True
                    else:
                        qmd.close()
                        qmd = None
                elif self.mode == "custom":
                    custom = CircAIdianLightRetrieval()
                    await custom.index_async(chunk_tuples, turns=scenario.turns)

            return cm, hyde, qmd, custom

    def _retrieve_bm25(self, cm: ActiveContextManager, query: str) -> str:
        chunks = cm.get_all_chunks()
        if not chunks:
            return ""
        chunk_tuples = [(c.chunk_id, c.content) for c in chunks]
        return bm25_retrieve(query, chunk_tuples, top_k=5)

    def _retrieve_qmd(self, cm: ActiveContextManager, query: str,
                      qmd: Optional[QMDRetrieval]) -> str:
        if qmd is None or not qmd._indexed:
            return self._retrieve_bm25(cm, query)
        chunks = cm.get_all_chunks()
        if not chunks:
            return ""
        chunk_tuples = [(c.chunk_id, c.content) for c in chunks]
        results = qmd.query(query, top_k=5)
        if not results:
            return bm25_retrieve(query, chunk_tuples, top_k=5)
        chunk_map = {cid: content for cid, content in chunk_tuples}
        retrieved = " ".join(chunk_map.get(cid, "") for cid, _ in results if cid in chunk_map)
        return retrieved if retrieved else bm25_retrieve(query, chunk_tuples, top_k=5)

    async def _retrieve_hyde(
        self, cm: ActiveContextManager, query: str, hyde: Optional[HYDEBM25Retrieval]
    ) -> str:
        if hyde is None:
            return self._retrieve_bm25(cm, query)
        chunks = cm.get_all_chunks()
        if not chunks:
            return ""
        return await hyde.query_async(query, top_k=5)

    async def _retrieve_custom(
        self, cm: ActiveContextManager, query: str, custom: Optional[CircAIdianLightRetrieval]
    ) -> str:
        if custom is None:
            return self._retrieve_bm25(cm, query)
        return await custom.query_async(query, top_k=5)

    async def _evaluate(
        self,
        cm: ActiveContextManager,
        hyde: Optional[HYDEBM25Retrieval],
        qmd: Optional[QMDRetrieval],
        custom: Optional[CircAIdianLightRetrieval],
        scenario: ConversationScenario,
    ) -> Dict:
        """Internal evaluation (caller manages retriever lifecycles)."""
        full_context = cm.get_context_for_prompt()
        full_tokens = cm.estimated_tokens

        # Retrieve context for each question
        retrieved_contexts = []
        for qa in scenario.qa_pairs:
            if self.mode == "bm25":
                ctx = self._retrieve_bm25(cm, qa.question)
            elif self.mode == "hyde":
                ctx = await self._retrieve_hyde(cm, qa.question, hyde)
            elif self.mode == "custom":
                ctx = await self._retrieve_custom(cm, qa.question, custom)
            else:  # qmd
                ctx = self._retrieve_qmd(cm, qa.question, qmd)
            retrieved_contexts.append(ctx)

        retrieved_tokens = sum(len(ctx.split()) for ctx in retrieved_contexts)

        # Score results
        disclosed_correct = 0
        disclosed_total = 0
        adversarial_correct_refusal = 0
        adversarial_total = 0
        details = []

        for qa, retrieved_ctx in zip(scenario.qa_pairs, retrieved_contexts):
            if qa.is_adversarial:
                adversarial_total += 1
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
                    result = "HALLUCINATED"
                details.append({"q": qa.question, "expected": qa.correct_answer, "result": result})
            else:
                disclosed_total += 1
                answer_keywords = {w.lower() for w in qa.correct_answer.split() if len(w) > 2}
                found = sum(1 for kw in answer_keywords if kw in retrieved_ctx.lower())
                if found >= len(answer_keywords) * 0.5:
                    disclosed_correct += 1
                    result = "CORRECT"
                else:
                    result = "INCORRECT"
                details.append({
                    "q": qa.question, "expected": qa.correct_answer,
                    "retrieved_snippet": retrieved_ctx[:100], "result": result,
                })

        memory_accuracy = disclosed_correct / disclosed_total if disclosed_total else 0
        adversarial_robustness = (
            adversarial_correct_refusal / adversarial_total if adversarial_total else 0
        )
        n_qa = len(scenario.qa_pairs)
        token_efficiency = max(0, 1 - (retrieved_tokens / (full_tokens * n_qa))) if full_tokens else 0

        return {
            "scenario_id": scenario.scenario_id,
            "persona": scenario.persona,
            "memory_accuracy": memory_accuracy,
            "adversarial_robustness": adversarial_robustness,
            "token_efficiency": token_efficiency,
            "disclosed_correct": disclosed_correct,
            "disclosed_total": disclosed_total,
            "adversarial_refused": adversarial_correct_refusal,
            "adversarial_total": adversarial_total,
            "full_context_tokens": full_tokens,
            "avg_retrieved_tokens": retrieved_tokens / n_qa if n_qa else 0,
            "details": details,
        }

    async def evaluate_scenario(self, scenario: ConversationScenario) -> Dict:
        cm, hyde, qmd, custom = await self.load_scenario(scenario)
        try:
            return await self._evaluate(cm, hyde, qmd, custom, scenario)
        finally:
            if hyde is not None:
                pass  # HYDE is stateless, no close needed
            if qmd is not None:
                qmd.close()
            if custom is not None:
                pass  # stateless

    async def run(self) -> Dict:
        print("=" * 60)
        print(f"CircAIdian Benchmark — {len(SCENARIOS)} scenarios | mode={self.mode}")
        print("=" * 60)
        print()

        all_results = []
        for scenario in SCENARIOS:
            print(f"  {scenario.scenario_id}...", end=" ", flush=True)
            result = await self.evaluate_scenario(scenario)
            all_results.append(result)
            print(
                f"MemAcc={result['memory_accuracy']:.0%} "
                f"AdvRob={result['adversarial_robustness']:.0%} "
                f"TokEff={result['token_efficiency']:.0%}"
            )

        # Aggregate
        n = len(all_results)
        avg_memory_acc = sum(r["memory_accuracy"] for r in all_results) / n
        avg_adversarial = sum(r["adversarial_robustness"] for r in all_results) / n
        avg_token_eff = sum(r["token_efficiency"] for r in all_results) / n
        total_disc = sum(r["disclosed_total"] for r in all_results)
        total_adv = sum(r["adversarial_total"] for r in all_results)
        total_disc_corr = sum(r["disclosed_correct"] for r in all_results)
        total_adv_ref = sum(r["adversarial_refused"] for r in all_results)

        print()
        print("=" * 60)
        print(f"OVERALL ({self.mode.upper()})")
        print("=" * 60)
        print(f"  Memory Accuracy:      {avg_memory_acc:.1%}  ({total_disc_corr}/{total_disc})")
        print(f"  Adversarial Robust:   {avg_adversarial:.1%}  ({total_adv_ref}/{total_adv})")
        print(f"  Token Efficiency:     {avg_token_eff:.1%}")

        return {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "max_context_tokens": self.max_context_tokens,
            "num_scenarios": n,
            "overall_memory_accuracy": avg_memory_acc,
            "overall_adversarial_robustness": avg_adversarial,
            "overall_token_efficiency": avg_token_eff,
            "total_disclosed_correct": total_disc_corr,
            "total_disclosed": total_disc,
            "total_adversarial_refused": total_adv_ref,
            "total_adversarial": total_adv,
            "per_scenario": [
                {k: r[k] for k in ("scenario_id", "memory_accuracy",
                                   "adversarial_robustness", "token_efficiency")}
                for r in all_results
            ],
        }


async def main():
    parser = argparse.ArgumentParser(description="CircAIdian LoCoMo benchmark")
    parser.add_argument(
        "--mode", choices=["bm25", "hyde", "custom", "qmd"], default="hyde",
        help="Retrieval mode: bm25 (baseline), hyde (HYDE+BM25), custom (M2.7 rerank), qmd (QMD hybrid)"
    )
    args = parser.parse_args()

    harness = MemoryBenchmarkHarness(max_context_tokens=4000, mode=args.mode)
    summary = await harness.run()

    out_path = Path(__file__).parent / f"benchmark_results_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
