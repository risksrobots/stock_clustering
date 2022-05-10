import numpy as np
import pandas as pd

class CalculatingStockFeatures:

    def calculate_stat(metric, pct_prices):
        """
        Returns a main describe statistics

        type : str, ['median', 'mean', 'mode', 'std']

        """

        if metric == 'mean':
            feature = np.mean(pct_prices)
        elif metric == 'median':
            feature = np.median(pct_prices)
        elif metric == 'std':
            feature = np.std(pct_prices)
        elif metric == 'mode':
            feature = np.mode(pct_prices)

        return feature


    def calculate_glob_max_drawdown(pct_prices_cumprod):
        """
        Returns a max drawdown in the whole period
        """
        return np.min(pct_prices_cumprod)


    def calculate_loc_max_drawdown(pct_prices_cumprod):
        """
        Returns a max drawdown after a max price
        """
        idx_max_price = np.argmax(pct_prices_cumprod)
        max_drawdown = np.min(pct_prices_cumprod[idx_max_price:])

        return max_drawdown


    def calculate_loc_max_price(pct_prices_cumprod):
        """
        Returns a max price after a max drawdown
        """
        idx_min_price = np.argmin(pct_prices_cumprod)
        max_price = np.max(pct_prices_cumprod[idx_min_price:])

        return max_price


    def calculate_pсt_days_of_profit_return(pct_prices):
        """
        Returns a percent of days where return is a plus value
        """
        profit_prices = 0

        for price in pct_prices:
            if price > 0:
                profit_prices += 1
        return profit_prices / len(pct_prices)


    def calculate_pct_days_after_max_drawdown(pct_prices):
        """
        Returns a percent of days after a max drawdown
        """
        idx_max_price = np.argmax(pct_prices)

        return len(pct_prices[idx_max_price:]) / len(pct_prices)


    def calculate_beta(pct_stock, pct_market):
        """
        Returns a ratio of covariance(stock, market) on variance(market)
        """
        beta = np.cov(pct_stock, pct_market)[0][1] / np.var(pct_market)
        return beta


    def calculate_alfa(pct_stock, pct_market, pct_stock_mean):
        """
        Returns an alfa: r_s_mean - b * r_m_mean
        """

        beta = CalculatingStockFeatures.calculate_beta(pct_stock, pct_market)
        alfa = pct_stock_mean - beta * np.mean(pct_market)

        return alfa


    def calculate_sharp(pct_prices, pct_stock_mean, wout_risk_asset=0.03**(1/252)):
        """
        Sharp: (R_s_mean - 0.03)/ std(R_s)
        """

        sharp = (pct_stock_mean - wout_risk_asset) / np.std(pct_prices)
        return sharp


    def calculate_VaR(pct_prices):
        """
        Returns a VaR: pct_change.quantile(0.05) = с вероятность 95 вы не просрете вот ету цифру
        """
        return np.quantile(pct_prices, 0.05)


    def calculate_CAPM(pct_stock, pct_stock_mean, pct_market, wout_risk_asset=0.03):
        """
        CAPM: безрисковая ставка + (R_m- безриск ставка) * B
        """

        beta = np.cov(pct_stock, pct_market)[0][1] / np.var(pct_market)
        capm = wout_risk_asset + (pct_stock_mean - wout_risk_asset) * beta

        return capm


    def calculate_IR(pct_stock, pct_stock_mean, pct_market):
        """
        IR: R_s_mean - R_m_mean/ std(R_s_mean - R_m_mean)
        """
        ir = pct_stock_mean - pct_market.mean() / np.std([i - j for i, j in zip(pct_stock, pct_market)])

        return ir


    def calculate_var_coef(pct_prices, pct_stock_mean):
        """
        Returns a coef of variation: std(pct_change)/ R_s_mean
        """
        var_coef = pct_prices.std() / pct_stock_mean

        return var_coef


    def calculate_cvar_price(pct_prices):
        """
        Returns a conditional var, CVAR price[price < Var].mean()
        """

        VaR = CalculatingStockFeatures.calculate_VaR(pct_prices)
        prices_var = []

        for price in pct_prices:
            if price < VaR:
                prices_var.append(price)

        return np.mean(prices_var)


    def calculate_rel_drawdown(prices, type_rel='entr'):
        """
        Drawdown related to enter: a pct change from enter point to max drawdown
        Drawdown related to high point: a pct change from high point to max drawdown

        type_rel: str, ['entr', 'max']
        """

        if type_rel == 'entr':
            drawdown = prices.min() / prices[0]
        elif type_rel == 'max':
            drawdown = prices.min() / prices.max()

        return drawdown


    def calculate_momentum_price(df_prices):
        """
        Returns a momentum price - range of stock's ratio values.
        """

        # calculating portfolio's ratios
        df_ratio = df_prices / df_prices.shift()
        # dropping first value due to it's NaN
        df_ratio_array = df_ratio.values[1:]
        # calculating range between max and min
        momentum_price = df_ratio_array.max() - df_ratio_array.min()

        return momentum_price


    def calculate_max_recovery_period(pct_prices):
        """
        legacy, took from @namelastname71
        Returns a num of days as a period of max recovery from loc max
        """
        max_price = pct_prices[0]
        curr_drawdown = 0
        max_drawdown = 0
        curr_left = 0
        left = 0
        right = 0
        for i in range(0, len(pct_prices)):
            curr_drawdown = pct_prices[i]
            if curr_drawdown < max_drawdown:
                max_drawdown = curr_drawdown
                left = curr_left
                right = i
            if pct_prices[i] > max_price:
                max_price = pct_prices[i]
                curr_left = i
        return right - left

    def calculate_features_for_stock(df_stock, df_market):
        prices = df_stock.values
        pct_stock = df_stock.pct_change()[1:].values
        pct_market = df_market.pct_change()[1:].values
        pct_stock_cumprod = (pct_stock + 1).cumprod()
        features = []


        # 1
        pct_stock_mean = np.mean(pct_stock)
        features.append(pct_stock_mean)

        # 2
        pct_stock_median = np.median(pct_stock)
        features.append(pct_stock_median)

        # 3
        pct_stock_std = np.std(pct_stock)
        features.append(pct_stock_std)

        # 4
        max_drawdown = CalculatingStockFeatures.calculate_glob_max_drawdown(pct_stock_cumprod)
        features.append(max_drawdown)

        # 5
        loc_max_drawdown = CalculatingStockFeatures.calculate_loc_max_drawdown(pct_stock_cumprod)
        features.append(loc_max_drawdown)

        # 6
        loc_max_price = CalculatingStockFeatures.calculate_loc_max_price(pct_stock_cumprod)
        features.append(loc_max_price)

        # 7
        pсt_days_of_profit_return = CalculatingStockFeatures.calculate_pсt_days_of_profit_return(pct_stock)
        features.append(pсt_days_of_profit_return)

        # 8
        pct_days_after_max_drawdown = CalculatingStockFeatures.calculate_pct_days_after_max_drawdown(pct_stock)
        features.append(pct_days_after_max_drawdown)

        # 9
        beta = CalculatingStockFeatures.calculate_beta(pct_stock, pct_market)
        features.append(beta)

        # 10
        alfa = CalculatingStockFeatures.calculate_alfa(pct_stock, pct_market, pct_stock_mean)
        features.append(alfa)

        # 11
        sharp = CalculatingStockFeatures.calculate_sharp(pct_stock, pct_stock_mean)
        features.append(sharp)

        # 12
        VaR = CalculatingStockFeatures.calculate_VaR(pct_stock)
        features.append(VaR)

        # 13
        capm = CalculatingStockFeatures.calculate_CAPM(pct_stock, pct_stock_mean, pct_market)
        features.append(capm)

        # 14
        ir = CalculatingStockFeatures.calculate_IR(pct_stock, pct_stock_mean, pct_market)
        features.append(ir)

        # 15
        var = CalculatingStockFeatures.calculate_var_coef(pct_stock, pct_stock_mean)
        features.append(var)

        # 16
        cvar = CalculatingStockFeatures.calculate_cvar_price(pct_stock)
        features.append(cvar)

        # 17
        rel_drawdown = CalculatingStockFeatures.calculate_rel_drawdown(prices)
        features.append(rel_drawdown)

        # 18
        momentum = CalculatingStockFeatures.calculate_momentum_price(df_stock)
        features.append(momentum)

        # 19
        max_recovery_period = CalculatingStockFeatures.calculate_max_recovery_period(pct_stock)
        features.append(max_recovery_period)

        assert (len(features) == 19)
        return features

    def calculate_features(df_stock_all, df_market):
        features_names = ['mean', 'median', 'std', 'glob_max_drawdown', 'loc_max_drawdown', 'loc_max_price',
                          'pct_days_of_profit_return', 'pct_days_after_max_drawdown',
                          'beta', 'alfa', 'sharp', 'VaR', 'CAPM', 'IR', 'variation_coef', 'CVaR', 'rel_drawdown',
                          'momentum', 'max_recovery_peiod']
        assert (len(features_names) == 19)
        results = pd.DataFrame(data=[], columns=features_names)

        for stock in df_stock_all.columns:
            df_stock = df_stock_all[stock]
            stock_features = CalculatingStockFeatures.calculate_features_for_stock(df_stock, df_market)
            results.loc[stock] = stock_features

        return results
