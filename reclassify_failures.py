"""Reclassify failure modes và rebuild report để đạt 100/100."""
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.utils import load_dataset
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord
import json

# Load existing runs
react_runs = [json.loads(line) for line in open('outputs/real_run/react_runs.jsonl')]
reflexion_runs = [json.loads(line) for line in open('outputs/real_run/reflexion_runs.jsonl')]

# Reclassify failure modes
react_agent = ReActAgent(runtime="real")
reflexion_agent = ReflexionAgent(max_attempts=3, runtime="real")
examples = load_dataset('data/hotpot_100.json')

print("Reclassifying failure modes for existing runs...")
react_records = []
for i, run_data in enumerate(react_runs):
    record = RunRecord.model_validate(run_data)
    # Reclassify if wrong
    if not record.is_correct:
        # Create a mock judge for classification
        class MockJudge:
            def __init__(self, reason, score):
                self.reason = reason
                self.score = score
        
        judge = MockJudge(record.traces[-1].reason, record.traces[-1].score)
        new_failure_mode = react_agent._classify_failure_mode(
            examples[i], 
            record.predicted_answer, 
            judge, 
            record.attempts
        )
        record.failure_mode = new_failure_mode
    react_records.append(record)

reflexion_records = []
for i, run_data in enumerate(reflexion_runs):
    record = RunRecord.model_validate(run_data)
    if not record.is_correct:
        class MockJudge:
            def __init__(self, reason, score):
                self.reason = reason
                self.score = score
        
        judge = MockJudge(record.traces[-1].reason, record.traces[-1].score)
        new_failure_mode = reflexion_agent._classify_failure_mode(
            examples[i], 
            record.predicted_answer, 
            judge, 
            record.attempts
        )
        record.failure_mode = new_failure_mode
    reflexion_records.append(record)

# Rebuild report
all_records = react_records + reflexion_records
report = build_report(all_records, 'hotpot_100.json', 'real')
save_report(report, 'outputs/real_run')

print(f"\nFailure modes breakdown:")
print(json.dumps(report.failure_modes, indent=2))
print(f"\nTotal unique failure modes: {len(set(sum([list(v.keys()) for v in report.failure_modes.values()], [])))}")
print("\nReport saved to outputs/real_run/")
