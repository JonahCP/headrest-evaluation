
from typing import List
import mne

def set_mapping(data, mapping: List[dict]) -> tuple:
    """
    Set mappings for events
    """
    events, event_id = mne.events_from_annotations(data)
    annot_from_events = mne.annotations_from_events(
    events=events,
    event_desc=mapping,
    sfreq=data.info['sfreq'],
    verbose='INFO'
    )
    data.set_annotations(annot_from_events)
    new_events, new_event_id = mne.events_from_annotations(data)
    return new_events, new_event_id