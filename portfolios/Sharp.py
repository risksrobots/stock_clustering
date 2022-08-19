import numpy as np
import random
import portfolios.sharp_derivative as functions_2


class SharpPortfolio:

    def __init__(self, returns, cov_mat, riskless_ret):
        self.returns = returns
        self.cov_mat = cov_mat
        self.riskless_ret = riskless_ret

    def fit(self, step=0.0015):
        random.seed(10)
        rs_list = [random.randint(0, 1000) for p in range(10 ** 4)]
        weights = np.array([1 / len(self.returns)] * len(self.returns))
        ret_risk_max = -10**7

        for i in range(10 ** 3):  # uniform or normal

            np.random.seed(rs_list[i])
            rand_vec = np.random.uniform(low=-step, high=0.015, size=self.returns.shape)
            rand_vec_centr = rand_vec - np.mean(rand_vec)

            weights_loc = weights + rand_vec_centr  # local  weights

            ret_loc, risk_loc = self.get_risk_ret(weights)
            ret_risk_loc = (ret_loc - self.riskless_ret) / risk_loc

            if ret_risk_loc > ret_risk_max and weights_loc.min() > 0.001 and weights_loc.max() < 0.5:
                weights = weights_loc
                ret_risk_max = ret_risk_loc

        first_method = ret_risk_max

        #ОПГ
        lr = 0.0000001
        num_of_iter = 100
        grad_weights = np.array([1 / len(self.returns)] * len(self.returns))
        weights_history = np.empty(num_of_iter, dtype=object)
        gradient_history = list()
        # ADAGRAD

        # MOMENTUM

        for _ in range(num_of_iter):
            function_value, gradient = functions_2.fAndG(self.cov_mat, self.returns, self.riskless_ret, grad_weights)
            weights_history[_] = grad_weights
            grad_weights = 2 * grad_weights - lr * gradient
            ret_loc, risk_loc = self.get_risk_ret(grad_weights)

        grad_weights = grad_weights / np.sum(grad_weights)
        ret_risk_loc = (ret_loc - self.riskless_ret) / risk_loc
        print(first_method, ret_risk_loc)

        if first_method < ret_risk_loc:
            weights = grad_weights

        return weights

    def get_risk_ret(self, weights):
        weights = weights.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)

        return_port = weights.T @ returns
        risk_port = np.sqrt(weights.T @ self.cov_mat @ weights)
        return [float(return_port), float(risk_port)]