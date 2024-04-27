# counter.py

class PersonCounter:
    def __init__(self):
        self.count = 0

    def increment(self, n=1):
        self.count += n

    def get_count(self):
        return self.count
