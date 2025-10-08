def generate_rolling_slope(groupby_column, df, window_size, target_column):
    """
    Calculate rolling slope using pandas groupby and rolling.
    
    :param groupby_column: Column to group by (e.g., run_id)
    :param df: Input DataFrame  
    :param window_size: Window size for rolling calculation
    :param target_column: Column to calculate slopes for
    :return: numpy array of rolling slopes
    """
    from scipy.stats import linregress
    import numpy as np
    
    def calc_slope(window):
        if len(window) < window_size:
            return np.nan
        return linregress(np.arange(len(window)), window).slope
    
    # Use transform to maintain original shape
    rolling_slopes = (
        df.groupby(groupby_column)[target_column]
        .rolling(window_size, min_periods=window_size)
        .apply(calc_slope, raw=True)
        .values
    )
    
    return rolling_slopes
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
