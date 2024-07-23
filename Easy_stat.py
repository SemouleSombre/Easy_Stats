import pandas as pd
import numpy as np
import scipy.stats as stats

N_max = 30
N_min = 5
Percentage_group_prop = 0.2
ALPHA_CHECK = 0.1


def Check_Normality(df:pd.Series)->bool:
    """
    Check if the series is normal
    """
    return stats.shapiro(df) > ALPHA_CHECK

def Check_Homogeneity(series:pd.Series)->bool:
    """
    Check if the series is homogenous
    """
    return stats.levene(series) > ALPHA_CHECK

def Check_Size(series:pd.Series)->bool:
    """
    Check if the series is big enough
    """
    return len(series) > N_max

def Check_Proportion_Group(list_proportion:list)->bool:
    """
    Check the condition for the proportion
    """
    size_by_group = []
    for proportion in list_proportion:
        if len(proportion) <= 0:
            return False
        else:
            size_by_group.append(len(proportion) < N_min)
    
    size_by_group = pd.Series(size_by_group)
    if size_by_group.mean() >= Percentage_group_prop:
        return False
    else:
        return True
    
def Check_Group(df:pd.DataFrame, function:callable)->bool:
    """
    Check if the series is big enough
    """
    try :
        function in [
            Check_Normality,
            Check_Homogeneity,
            Check_Size
        ]
    except ValueError:
        print("Error : function not found")
        
    try:
        if len(df.columns) <= 0:
            raise ValueError("Error : df is empty")
    except ValueError as e:
            print(e)

    
    for column in df.columns:
        df_column = df[column]
        if function(df_column) == False:
            return False
    return True

def Check_Side(side:str)->bool:
    """
    Check if the side is correct
    """
    try:
        side in ['Two', 'Left', 'Right']
    except ValueError:
        print("Error : side not found")
    return True

def Compare_Mean(df:pd.DataFrame, columns:list, size:int ,alpha:float=0.05, side:str='Two', ref:float=None, dependant:float=True)->float:
    """
    Compare the mean of a Value, Series or a DataFrame
    """
    try:
        if size == 1:
            return Compare_Mean_to_Ref_Value(df=df, columns=columns, alpha=alpha, side=side, ref=ref)   
        elif size == 2:
            return Compare_two_Means(df=df, columns=columns, alpha=alpha, side=side, dependant=dependant)
        elif size > 2:
            return Compare_Mean_to_Ref_Group(df=df, columns=columns, alpha=alpha, side=side)
        else:
            raise ValueError("Error : size must be greater than 0")
    except ValueError as e:
        print(e)

def Compare_Mean_to_Ref_Value(df:pd.DataFrame, columns:list, alpha:float, side:str, ref:float)->bool:
    try:
        if len(columns) != 1:
            raise ValueError("Error : columns must be a list with one element")
    except ValueError as e:
        print(e)
    
    try:
        if type(ref) != float:
            raise ValueError("Error : ref must be a float")
    except ValueError as e:
        print(e)
        
    try:
        if Check_Side(side) == False:
            raise ValueError("Error : side must be a string in ['Two', 'Left', 'Right']")
    except ValueError as e:
        print(e)
        
    try:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Error : alpha must be between 0 and 1 advice 0.05")
    except ValueError as e:
        print(e)
        
    if not Check_Size(df[columns[0]]) and not Check_Normality(df[columns[0]]):
        print("Running Non-Parametric Wilcoxon Rank Sum Test")
        p = 1
    else:
        print("1 Sample T-test")
        p = 1
    return p < alpha
        
def Compare_two_Means(df:pd.DataFrame, columns:list, alpha:float, side:str, dependant:float)->float:
    
    pass


def Compare_Mean_to_Ref_Group(df:pd.DataFrame, columns:list, alpha:float, side:str, ref:list, dependant:float)->float:
    pass