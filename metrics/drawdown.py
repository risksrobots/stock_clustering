import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

class Drawdown_plot:

    def __init__(self, ports_pct, start_price_of_port, label='imoex'):
        self.ports_pct = ports_pct
        self.start_price = start_price_of_port
        self.label = label

    def find_max_drawdown(self, prices):
        """
        Takes Series with closing prices.
        Returns the value of maximum drawdown
        in percent and indexes of prices where this
        maximum drawdown took place. If stock is
        always growing it will return minimum
        growth with and indexes of prices where this
        minimum growth took place.
        """
        max_price = prices.iloc[0]
        curr_drawdown = 0
        max_drawdown = 0
        curr_left = 0
        left = 0
        right = 0
        for i in range(0, len(prices)):
            curr_drawdown = (prices.iloc[i] / max_price - 1) * 100
            # print(type(curr_drawdown))
            if curr_drawdown < max_drawdown:
                max_drawdown = curr_drawdown
                left = curr_left
                right = i
            if prices.iloc[i] > max_price:
                max_price = prices.iloc[i]
                curr_left = i
        return max_drawdown, left, right

    def calc_growth(self, prices):
        """
        Calculates list with growth
        """
        growth = []
        past_p = 0
        for p in prices:
            if past_p:
                growth.append(p - past_p)
            past_p = p
        return growth

    def find_max_recovery(self, prices):
        """
        Takes Series with closing prices.
        Returns the value of maximum recovery
        period in days and indexes of prices
        where this recovery period took place.
        """
        growth = self.calc_growth(prices)
        s = 0
        left = 0
        right = 0
        curr_left = 0
        max_recovery = 0
        for i in range(0, len(growth)):
            if not s:
                curr_left = i
            s += growth[i]
            if s > 0:
                s = 0
                if max_recovery < (i - curr_left):
                    max_recovery = i - curr_left
                    left = curr_left
                    right = i

        return max_recovery, left, right + 1

    def show_stats_of_stock(self, data):

        """
        draw a plot with
        - max drawdown for whole period
        - max period of recovery for whole period
        на вход поступает pandas Series, индекс=датам, значения=ценам портфеля
        """

        # find max drowdawn boundaries in appropriate format
        max_drawdown = self.find_max_drawdown(data)
        max_drawdown_date = [pd.to_datetime(data.axes[0].tolist()[max_drawdown[1]]),
                             pd.to_datetime(data.axes[0].tolist()[max_drawdown[2]])]
        max_drawdown_val = [data.iloc[max_drawdown[1]], data.iloc[max_drawdown[2]]]

        # find max recovery period boundaries in appropriate format
        max_recovery = self.find_max_recovery(data)
        max_recovery_date = [pd.to_datetime(data.axes[0].tolist()[max_recovery[1]]),
                             pd.to_datetime(data.axes[0].tolist()[max_recovery[2]])]
        max_recovery_val = [data.iloc[max_recovery[1]], data.iloc[max_recovery[2]]]

        # plot data
        fig, ax = plt.subplots(1, figsize=(15, 7))
        plt.plot(data, color='blue')

        # plot max drawdowd
        max_drawdown_section = np.arange(max_drawdown[1], max_drawdown[2], 1)
        date_max_drawdown_section = pd.to_datetime(np.array(data.axes[0].tolist())[max_drawdown_section])
        # plt.fill_between(date_max_drawdown_section, data[date_max_drawdown_section], color = 'red', label = "макс. просадка")
        # print(data.iloc[max_drawdown[1]])
        plt.hlines(data.iloc[max_drawdown[1]], data.index[0], data.index[max_drawdown[1]], label="max drawdown",
                   colors='#808080', linestyles='--', linewidth=2)
        plt.hlines(data.iloc[max_drawdown[2]], data.index[0], data.index[max_drawdown[2]], colors='#808080',
                   linestyles='--', linewidth=2)

        plt.ylabel("max drawdown {}%".format(round(-max_drawdown[0], 1)), fontweight='bold', fontsize=11)

        # plot max recovery period
        max_recovery_section = np.arange(max_recovery[1], max_recovery[2], 1)
        date_max_recovery_section = pd.to_datetime(np.array(data.axes[0].tolist())[max_recovery_section])
        max_price = max(data)
        # plt.fill_between(date_max_recovery_section, data[date_max_recovery_section], max_price + 1, color = 'magenta', label = "макс. период восстановления")
        plt.fill_between(date_max_recovery_section, data[date_max_recovery_section], color='#ffc0cb',
                         label="max recovery per.")

        plt.xlabel("max recovery per {} days".format(round(max_recovery[0], 0)), fontweight='bold', fontsize=11)

        xfmt = mdates.DateFormatter('%m,%Y')
        ax.xaxis.set_major_formatter(xfmt)

        plt.ylim([min(data) - 0.1, np.max(data.values) + 0.1])

        fig.patch.set_visible(False)
        # ax.axis('off')

        plt.grid(color='#cccccc', linewidth=0.5)
        plt.title(self.label)
        plt.legend()
        plt.show()

    def plot(self):
        percent = self.ports_pct[self.label].values + 1
        arr = []
        for p in percent:
            self.start_price *= p
            arr.append(self.start_price)

        ser = pd.Series(arr)
        ser.index = self.ports_pct.index
        self.show_stats_of_stock(ser)

#ports_pct = pd.read_csv(f'results/df_all_ports_pct.csv', index_col=0)
#ports_pct.index = pd.to_datetime(ports_pct.index)
#ports_pct = ports_pct.rename(columns={'IMOEX_pct_change':'IMOEX', 'TP_mark_pct_change':'Markov',
#                                                    'TP_tobin_pct_change':'Tobin', 'TP_sharp_pct_change':'Sharp'})

#test = Drawdown_plot(ports_pct, 1, 'Markov')
#test.plot()