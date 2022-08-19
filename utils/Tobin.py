from scipy.optimize import linprog
import numpy as np

class TobinPortfolio:

    def __init__(self, returns, cov_mat, riskless_ret, imoex_risk):
        self.returns = returns
        self.cov_mat = cov_mat
        self.riskless_ret = riskless_ret
        self.imoex_risk = imoex_risk

        self.w_riskless = 0.000001
        self.w = np.matrix([0.01]*len(self.returns))
        self.risk = (self.w).dot(self.cov_mat.dot(self.w.T))
        self.ret = 0


    def ret(self):
        self.ret = 0
        for i in range(len(self.returns)):
            self.ret += self.returns[i]*np.array(self.w)[0][i]
        self.ret+=self.w_riskless*self.riskless_ret
        return self.ret

    # Здесь решается задача линейного программирования. Максимизируем доходность при риске не выше риска IMOEX за этот период.
    '''
    linprog() решает только задачи минимизации (не максимизации) и не допускает ограничений-неравенств со знаком 
    больше или равно (≥). Чтобы обойти эти проблемы, нам необходимо изменить описание задачи перед запуском оптимизации:
    Вместо максимизации доходности минимизируем отрицательное значение.
    '''
    def fit(self):
        obj = self.returns*(-1)
        obj -= self.riskless_ret
        #       └┤ Коэффициенты


        len_w = [1]*len(self.returns)

        lhs_ineq = [len_w,  # левая сторона красного неравенства
                    np.array(np.transpose(self.cov_mat.dot(self.w.T)))[0]]  # левая сторона синего неравенства

        rhs_ineq = [1,  # правая сторона красного неравенства
                    self.imoex_risk]  # правая сторона синего неравенства

        opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, method="revised simplex")

        return opt.x



