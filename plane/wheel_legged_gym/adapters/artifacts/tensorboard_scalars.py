"""Read selected TensorBoard tags without importing simulator or learning code."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_scalars(run, keys):
    events = EventAccumulator(str(run), size_guidance={'scalars': 0})
    events.Reload()
    available = events.Tags()['scalars']
    return {key: [(event.step, event.value) for event in events.Scalars(key)]
            for key in keys if key in available}
