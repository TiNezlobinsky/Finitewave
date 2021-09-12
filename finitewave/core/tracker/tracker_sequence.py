

class TrackerSequence:
    def __init__(self):
        self.sequence = []
        self.model = None

    def initialize(self, model):
        self.model = model
        for tracker in self.sequence:
            tracker.initialize(model)

    def add_tracker(self, tracker):
        self.sequence.append(tracker)

    def remove_trackers(self):
        self.sequence = []

    def tracker_next(self):
        for tracker in self.sequence:
            tracker.track()
