from scipy.optimize import minimize
import numpy as np
import pandas as pd
from datetime import datetime


class MarkowitzPortfolio:
    def __init__(self, returns, cov_mat, ret_det=None, args=None):
        self.returns = returns
        self.cov_mat = cov_mat
        if ret_det is not None:
            self.ret_det = ret_det
        elif args is not None and 'ret_det' in args.keys():
            self.ret_det = args['ret_det']
        else:
            raise ValueError('ret_det or args must be set')

    def fit(self, nonneg_weights=True):

        def objective(x):  # функция риска
            return np.array(x).T @ self.cov_mat @ np.array(x)

        def constraint1(x):  # условие для суммы долей -1
            return 1.0 - np.sum(np.array(x))

        def constraint2(x):  # задание доходности
            return self.returns.T @ np.array(x) - self.ret_det

        n = len(self.returns)
        x0 = [1 / n] * n  # начальное значение переменных для поиска минимума функции риска
        b = (0.0, 0.3)  # условие для  x от нуля до единицы включая пределы
        bnds = [b] * n  # передача условий в функцию  риска(подготовка)
        con1 = {'type': 'ineq', 'fun': constraint1}  # передача условий в функцию  риска(подготовка)
        con2 = {'type': 'ineq', 'fun': constraint2}  # передача условий в функцию  риска(подготовка)
        cons = [con1, con2]  # передача условий в функцию  риска(подготовка)
        sol = minimize(objective, x0, method='SLSQP', \
                       bounds=bnds, constraints=cons)
        # print(prob.status)

        weights = sol.x
        return weights


def backtesting_universal(data, port_model=MarkowitzPortfolio, **args):
    weights_year = []
    return_portfolio = pd.DataFrame([])
    window_size = 2
    train_start_year = 2018
    test_start_year = train_start_year + window_size

    for i in range(8):  # цикл, при помощи которого ты скользящее среднее
        year = i // 4
        train_year = train_start_year + year
        test_year_start = test_start_year + year
        test_year_end = test_year_start

        month_train = 1 + i % 4 * 3

        month_test = month_train + 3
        if month_test > 12:
            test_year_end += 1
            month_test = month_test % 12
        # фильтрация данных
        returns_train = data[(data.index > datetime(train_year, month_train, 1)) &
                             (data.index < datetime(train_year + window_size, month_train, 1))]

        mu = (((returns_train + 1).prod()) ** (
                    1 / len(returns_train)) - 1).values * 252  # средняя доходность за год (252 раб дня)
        Sigma = returns_train.cov().values * 252  # ковариационная матрица за год (252 раб дня)

        port_ = port_model(mu, Sigma, args=args)
        weights = port_.fit()

        weights_year.append(weights)

        returns_test = data[(data.index > datetime(test_year_start, month_train, 1)) &
                            (data.index < datetime(test_year_end, month_test, 1))]

        # расчет динамики портфеля за данный период
        return_portfolio_loc = pd.DataFrame(returns_test.values @ weights, index=returns_test.index,
                                            columns=['portfolio'])

        # запись результатов динамики в результирующую переменную
        return_portfolio = pd.concat([return_portfolio, return_portfolio_loc])

    return weights_year, return_portfolio