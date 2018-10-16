import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

class SoftmaxClassifier():
    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.output_layer = nn.Linear(input_size, output_size)

        def forward(self, x):
            return self.output_layer(x)

    def __init__(self, input_size, output_size, weight_decay, logger=None):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.model = self.Model(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=weight_decay)
        self.logger = logger

    def _set_learning_rate(self, new_learning_rate):
        for group in self.optimizer.param_groups:
            group['lr'] = new_learning_rate

    def train(self, batched_data, batch_size, num_epochs, learning_rate=0.001, num_logs=10):

        total_steps = num_epochs * (len(batched_data))
        log_every_n = total_steps // num_logs
        current_step = 1

        self._set_learning_rate(learning_rate)

        for epoch in range(num_epochs):
            
            correct = 0
            running_loss = 0.0
            for batch in batched_data:
                self.optimizer.zero_grad()

                data = batch['data']
                labels = torch.squeeze(batch['label'])

                output = self.model(data)
                loss = F.cross_entropy(input=output, target=labels)
                
                loss.backward()
                self.optimizer.step()

                _, predicted_labels = torch.max(output, 1)
                correct+= (predicted_labels == labels).sum().item()

                running_loss += loss.item()

                if current_step == 1:
                    if self.logger:
                        self.logger.info('Starting loss: {0:.8f} accuracy: {1:.2f}'.format(running_loss, correct/batch_size))

                if current_step % log_every_n == 0:

                    avg_loss = running_loss/log_every_n
                    avg_acc = correct/(log_every_n * batch_size)

                    if self.logger: 
                        self.logger.info('At step {0:} in epoch {1}: loss: {2:.8f} accuracy: {3:.2f}'.format(current_step, epoch, avg_loss, avg_acc))

                    running_loss = 0.0
                    correct = 0
                
                current_step+=1

    def evaluate(self, batched_data):
        correct = 0
        running_loss = 0.0
        
        total_batches = len(batched_data)
        batch_size = 0
        for batch in batched_data:
            batch_size = len(batch['data'])

            data = batch['data']
            labels = torch.squeeze(batch['label'])

            with torch.set_grad_enabled(False):
                output = self.model(data)
                loss = F.cross_entropy(input=output, target=labels)

            _, predicted_labels = torch.max(output, 1)
            correct+= (predicted_labels == labels).sum().item()

            running_loss += loss.item()

        return running_loss/total_batches, correct/(total_batches * batch_size)
        

if __name__ == "__main__":
    pass


            
