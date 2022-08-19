import numpy as np
import random


class QSharpPortfolio:

    def __init__(self, assets_returns, returns, cov_mat, riskless_ret, imoex_risk):
        self.assets_returns = assets_returns
        self.returns = returns
        self.cov_mat = cov_mat
        self.riskless_ret = riskless_ret
        self.w = np.array([0.01] * len(self.returns))
        self.imoex_risk = imoex_risk

    def get_params(self):
        ret_asset = []
        for i in range(len(self.assets_returns.columns)):
            ret_asset.append(self.assets_returns.iloc[:, i].values.mean())
        ret_single_port = []
        for i in range(len(self.assets_returns)):
            ret_single_port.append(self.assets_returns.iloc[i, :].values.mean())

        ret_asset = np.array(ret_asset)
        ret_single_port = np.array(ret_single_port)
        mean_ret_single_port = ret_single_port.mean()
        denominator = (ret_single_port-mean_ret_single_port)**2
        denominator_sum = sum(denominator)

        all_numerators = []
        numerators_sum = []
        beta = []
        for i in range(len(self.assets_returns.columns)):
            numerator = (self.assets_returns.iloc[:, i].values-ret_asset[i])*(ret_single_port-mean_ret_single_port)
            numerators_sum.append(sum(numerator))
            beta.append(sum(numerator)/denominator_sum)
            all_numerators.append(numerator)

        all_residual_risks = []
        mean_residual_risks = []
        for i in range(len(self.assets_returns.columns)):
            residual_risk = (self.assets_returns.iloc[:, i].values-ret_asset[i]-beta[i]*all_numerators[i])**2
            all_residual_risks.append(residual_risk)
            mean_residual_risks.append(residual_risk.mean())

        risk_single_port = (denominator_sum/len(denominator))**(0.5)
        rsp = mean_ret_single_port-ret_single_port[-1] #Ром, как эту штуку назвать?

        return ret_asset, np.array(beta), np.array(mean_residual_risks), risk_single_port, rsp


    def get_risk_ret(self, w):

        ret_asset, beta, mean_residual_risks, risk_single_port, rsp = self.get_params()
        r_w = ret_asset*w
        b_w = beta*w
        b_w_2 = b_w**2
        sigma2_w2 = (mean_residual_risks**2)*(w**2)
        ret = sum(ret_asset*w) + sum(rsp*beta*w)
        risk = ((risk_single_port**2)*sum(b_w_2)+sum(sigma2_w2))**0.5

        return ret, risk


    def fit(self, step=0.0015):

        random.seed(10)
        rs_list = [random.randint(0, 1000) for p in range(10 ** 4)]
        weights = np.array([1 / len(self.returns)] * len(self.returns))
        ret_max = -1

        for i in range(10 ** 3):  # uniform or normal

            np.random.seed(rs_list[i])
            rand_vec = np.random.uniform(low=-step, high=0.015, size=self.returns.shape)
            rand_vec_centr = rand_vec - np.mean(rand_vec)

            weights_ = weights + rand_vec_centr  # local  weights
            weights_loc = []

            for weight in weights_:
                if weight<0:
                    weights_loc.append(0)
                else:
                    weights_loc.append((weight))

            weights_loc = np.array(weights_loc)
            print(weights_loc)

            ret_loc, risk_loc = self.get_risk_ret(weights)

            if ret_loc >= ret_max and risk_loc <= self.imoex_risk and sum(weights_loc)<=1:
                weights = weights_loc
                ret_max = ret_loc

        return weights


