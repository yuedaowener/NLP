import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

def remove_duplicate_ref(params, grads):
    '''
    将参数列表中重复的权重整合为1个，
    加上与该权重对应的梯度
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 在共享权重的情况下
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 在作为转置矩阵共享权重的情况下（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

def clip_grads_ref(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


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

                loss = model.forward(x_batch, t_batch)
                model.backward()
                # 在这里params和model.params指向的是同一个对象。 id(model.params[0]) == id(params[0]): True
                params, grads = remove_duplicate_ref(model.params, model.grads)
                if max_grad is not None:
                    clip_grads_ref(grads, max_grad)
                optimizer.update(params, grads)
                loss_total += loss
                loss_count += 1

                if eval_interval and (batch_id % eval_interval == 0):
                    loss_avg = loss_total / loss_count
                    elapsed_time = time.time() - start_time
                    print(f"epoch_id:{epoch_id}/{max_epoch}, batch_id:{batch_id} /{n_batch}, time={elapsed_time}, loss_avg={loss_avg}")
                    loss_total = 0
                    loss_count = 0
                    self.loss_list.append(loss_avg)


        pass

    def plot(self, fig_save_path="loss_curve.png"):
        loss_list = np.array(self.loss_list)
        plt.plot(loss_list)
        plt.title("loss curve")
        plt.xlabel("iter time")
        plt.ylabel("loss")
        # plt.show()
        plt.savefig(fig_save_path)
        plt.close()
        print("loss curve saved as: ", fig_save_path)
        pass
