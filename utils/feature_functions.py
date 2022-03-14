
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
