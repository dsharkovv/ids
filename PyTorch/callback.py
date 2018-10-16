class EstimatorCallback():

    def __init__(self):
        super().__init__()

    def before_run(self):
        pass

    def on_run_end(self):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_start(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def before_evaluation(self, *args):
        pass