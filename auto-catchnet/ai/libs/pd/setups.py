import pandas as pd


def set_natural_notation():
    pd.options.display.float_format = '{:20,.2f}'.format
