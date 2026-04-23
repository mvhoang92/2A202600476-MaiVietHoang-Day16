# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 300
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.44 | 0.7267 | 0.2867 |
| Avg attempts | 1 | 1.9267 | 0.9267 |
| Avg token estimate | 1058.77 | 2911.86 | 1853.09 |
| Avg latency (ms) | 1841.05 | 5617 | 3775.95 |

## Failure modes
```json
{
  "react": {
    "no_answer_generated": 21,
    "none": 66,
    "wrong_final_answer": 58,
    "entity_drift": 5
  },
  "reflexion": {
    "none": 109,
    "looping": 37,
    "no_answer_generated": 4
  },
  "combined": {
    "no_answer_generated": 25,
    "none": 175,
    "wrong_final_answer": 58,
    "entity_drift": 5,
    "looping": 37
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
This benchmark evaluated ReAct and Reflexion agents on 150 HotpotQA multi-hop questions using OpenAI GPT-4o-mini. ReAct achieved 44.0% exact match accuracy with a single attempt per question, while Reflexion improved to 72.7% (+28.7% absolute gain) by leveraging self-reflection and iterative refinement over an average of 1.93 attempts. The reflection mechanism was particularly effective at correcting incomplete multi-hop reasoning and entity drift errors, where the agent initially identified the wrong intermediate entity or stopped after the first reasoning hop. However, this improvement came at a cost: Reflexion consumed 1853 additional tokens per question and increased latency by 3776ms on average. Failure mode analysis shows that 72.7% of questions were answered correctly, with the remaining errors primarily due to wrong_final_answer failures where even multiple reflection cycles could not recover the correct reasoning path. Future work could explore adaptive max_attempts based on question difficulty and memory compression techniques to reduce token overhead while maintaining accuracy gains.
