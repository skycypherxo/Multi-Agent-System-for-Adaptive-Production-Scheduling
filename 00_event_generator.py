import random
import time

class EventGenerator:
    def __init__(self):
        self.event_id = 0
        self.event_queue = []

    def generate_event(self):
        """
        Randomly generate a new job event, either normal or rush order,
        and add it to the event queue.
        """
        self.event_id += 1
        job_id = f"J{self.event_id:04d}"

        # Randomly decide if this is a rush order (20% chance)
        is_rush = random.random() < 0.2

        event = {
            "event_type": "new_order",
            "job_id": job_id,
            "priority": "high" if is_rush else "normal",
            "due_date": random.randint(5, 20) if is_rush else random.randint(15, 40),
            "processing_time": random.randint(1, 5)
        }

        self.event_queue.append(event)

    def get_event(self):
        """
        Return the next event from the queue, or None if empty.
        """
        if self.event_queue:
            return self.event_queue.pop(0)
        return None

    def simulate(self, steps=10, event_chance=0.5):
        """
        Simulate event generation over a number of steps.
        On each step, generate an event with probability event_chance.
        """
        for _ in range(steps):
            if random.random() < event_chance:
                self.generate_event()
            # For demonstration: print current queue
            print(f"Step {_+1}: Queue length = {len(self.event_queue)}")
            time.sleep(0.5)  # simulate time passing

if __name__ == "__main__":
    generator = EventGenerator()
    generator.simulate()
