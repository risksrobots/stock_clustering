import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def mean_year_return(df):
    year_return = df.reset_index().groupby(pd.Grouper(key="Date", freq="Y")).sum() + 1
    return year_return.prod() ** (1 / len(year_return)) - 1


def mean_year_vol(df):
    return df.reset_index().groupby(pd.Grouper(key="Date", freq="Y")).std().apply(lambda x: x * 252 ** 0.5).mean()


def find_max_drawdown(prices):
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
        if curr_drawdown < max_drawdown:
            max_drawdown = curr_drawdown
            left = curr_left
            right = i
        if prices.iloc[i] > max_price:
            max_price = prices.iloc[i]
            curr_left = i
    return max_drawdown, left, right


def calc_growth(prices):
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


def find_max_recovery(prices):
    """
    Takes Series with closing prices.
    Returns the value of maximum recovery
    period in days and indexes of prices
    where this recovery period took place.
    """
    growth = calc_growth(prices)
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


def plot_weights_pie(weights_year, assets, label=''):
    ress = []

    for j, weight in enumerate(weights_year):
        as_wei = dict()

        for i in range(len(assets)):
            as_wei[assets[i]] = weight[i]

        res = pd.DataFrame(as_wei.values(), index=as_wei.keys(), columns=['Asset']).query('Asset > 0.02')
        res = res.sort_values(by='Asset')
        if round(1 - res.Asset.sum(), 5) != 0:
            res.loc['others'] = round(1 - res.Asset.sum(), 5)

        ress.append(res)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 9))

    fig.suptitle(f'** {label} **', fontsize=18)
    for i, ax in enumerate(axes.flat):
        labels = ress[i].index
        sizes = ress[i].values

        colors = [plt.cm.Spectral(i / float(len(sizes.flatten()) - 1)) for i in range(len(sizes))]
        ax.pie(sizes.flatten(), labels=labels,
               autopct='%1.1f%%', shadow=True, startangle=140, colors=colors)

        ax.set_title(str(int(2018 + i)), {'fontsize': 19})

    plt.show()

def calculate_measures(return_portfolio):
    """
    return a list with:
    - mean year return
    - mean year volatility
    - max drawdown for whole period
    - max period of recovery for whole period
    """

    mean_year_ret = mean_year_return(return_portfolio['portfolio'])
    mean_year_risk = mean_year_vol(return_portfolio['portfolio'])

    max_drawdown = find_max_drawdown(return_portfolio['cumprod'])
    max_recovery = find_max_recovery(return_portfolio['cumprod'])

    port_measures = list(map(float, [mean_year_ret, mean_year_risk, max_drawdown[0], max_recovery[0]]))

    return port_measures


def show_drawdown_recovery(data, label=''):
    """
    draw a plot with
    - max drawdown for whole period
    - max period of recovery for whole period
    """

    # find max drowdawn boundaries in appropriate format
    max_drawdown = find_max_drawdown(data)

    # find max recovery period boundaries in appropriate format
    max_recovery = find_max_recovery(data)

    # plot data
    fig, ax = plt.subplots(1, figsize=(8, 4.3))
    plt.plot(data, color='blue')

    # plot max drawdowd
    plt.hlines(data.iloc[max_drawdown[1]], data.index[0], data.index[max_drawdown[1]], label="max drawdown",
               colors='#808080', linestyles='--', linewidth=2)
    plt.hlines(data.iloc[max_drawdown[2]], data.index[0], data.index[max_drawdown[2]], colors='#808080',
               linestyles='--', linewidth=2)

    plt.ylabel("max drawdown {}%".format(round(-max_drawdown[0], 1)), fontweight='bold', fontsize=11)

    # plot max recovery period
    max_recovery_section = np.arange(max_recovery[1], max_recovery[2], 1)
    date_max_recovery_section = pd.to_datetime(np.array(data.axes[0].tolist())[max_recovery_section])
    plt.fill_between(date_max_recovery_section, data[date_max_recovery_section], color='#ffc0cb',
                     label="max recovery per.")

    plt.xlabel("max recovery per {} days".format(round(max_recovery[0], 0)), fontweight='bold', fontsize=11)

    xfmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(xfmt)

    plt.ylim([min(data) * 0.99, np.max(data.values) * 1.01])

    fig.patch.set_visible(False)

    plt.grid(color='#cccccc', linewidth=0.5)
    plt.title(label)
    plt.legend()
    plt.show()