"""Read latest TensorBoard scalars"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator("checkpoints/runs")
ea.Reload()
scalars = {}
tags = ea.Tags().get("scalars", [])
if tags:
    first_tag = tags[0]
    events = ea.Scalars(first_tag)
    print(f"Tag: {first_tag}")
    print(f"Total steps: {len(events)}")
    print()

for tag in tags:
    events = ea.Scalars(tag)
    if events:
        step = events[-1].step
        val = events[-1].value
        scalars[tag] = (step, val)

for k, (step, val) in sorted(scalars.items()):
    print(f"  step={step:>8}  {k:<30} {val:.8f}")
