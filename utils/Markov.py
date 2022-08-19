from scipy.optimize import minimize
import numpy as np

class MarkovPortfolio:
    def __init__(self, returns, cov_mat, ret_det=None):
        self.returns = returns
        self.cov_mat = cov_mat
        if ret_det is not None:
            self.ret_det = ret_det

    def fit(self):

        def objective(x):  # функция риска
            return np.array(x).T @ self.cov_mat @ np.array(x)

        def constraint1(x):  # условие для суммы долей -1
            return 1.0 - np.sum(np.array(x))

        def constraint2(x):  # задание доходности
            return self.returns.T @ np.array(x) - self.ret_det

        n = len(self.returns)
        x0 = [1/n]*n  # начальное значение переменных для поиска минимума функции риска
        b = (0.0, 0.3)  # условие для  x от нуля до единицы включая пределы
        bnds = [b] * n  # передача условий в функцию  риска(подготовка)
        con1 = {'type': 'ineq', 'fun': constraint1}  # передача условий в функцию  риска(подготовка)
        con2 = {'type': 'ineq', 'fun': constraint2}  # передача условий в функцию  риска(подготовка)
        cons = [con1, con2]  # передача условий в функцию  риска(подготовка)
        sol = minimize(objective, x0, method='SLSQP', \
                       bounds=bnds, constraints=cons)

        weights = sol.x
        return weights

    def port_return(self, weights, assets_pct):

        pass