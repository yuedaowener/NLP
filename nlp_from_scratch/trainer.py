import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

class Trainer:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
        pass

    def fit(self, x, t, max_epoch=10, batch_size = 32, max_grad = None, eval_interval=20, verbose=True):
        n_sample = len(x)
        n_batch = n_sample // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        loss_total = 0
        loss_count = 0

        start_time = time.time()
        for epoch_id in range(max_epoch):
            idx = np.random.permutation(np.arange(n_sample))
            x = x[idx]
            t = t[idx]

            for batch_id in range(n_batch):
                x_batch = x[batch_id *batch_size : (batch_id + 1) * batch_size]
                t_batch = t[batch_id *batch_size : (batch_id + 1) * batch_size]

                loss = model.forward(x_batch, t_batch) # 因为还要计算loss
                model.backward()
                params, grads = model.params, model.grads
                optimizer.update(params, grads)
                loss_total += loss
                loss_count += 1


                if eval_interval and (batch_id % eval_interval == 0):   # 打印迭代是在bach 里面的？
                    loss_avg = loss_total / loss_count
                    elapsed_time = time.time() - start_time
                    print(f"epoch_id:{epoch_id}/{max_epoch}, batch_id:{batch_id} /{n_batch}, time={elapsed_time}, loss_avg={loss_avg}")
                    loss_total = 0
                    loss_count = 0
                    self.loss_list.append(loss_avg)


        pass

    def plot(self):
        loss_list = np.array(self.loss_list)
        plt.plot(loss_list)
        plt.xlabel("iter time")
        plt.ylabel("loss")
        plt.show()
        pass


