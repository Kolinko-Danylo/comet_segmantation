class TrainingMonitor:
    def __init__(self, interval):
        self.counter = 0
        self.interval = interval
        self.loss = None

    def reset(self):
        self.counter = 0

    def update(self, add_value=1):
        self.counter += add_value

    def update_best_model(self, loss):
        if self.loss is None or loss < self.loss:
            self.loss = loss
            return True
        return False

    def should_save_checkpoint(self):
        return self.counter >= self.interval

    @classmethod
    def from_config(cls, monitor_config):
        return cls(interval=monitor_config['interval'])