import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
from statsmodels.stats.contingency_tables import mcnemar

N_max = 30
N_min = 5
Percentage_group_prop = 0.2
ALPHA_CHECK = 0.1

# Configuration du logger
logging.basicConfig(level=logging.INFO)

def Define_Verbose(debug:bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug mode activated")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Debug mode deactivated")
    
def Check_Normality(series:pd.Series)->bool:
    """
    Check if the series is normal
    """
    statistic, p_val = stats.shapiro(series)
    logging.debug(f"Statistics : {statistic}")
    logging.debug(f"p value : {p_val}")
    test = p_val > ALPHA_CHECK
    logging.debug(f"Validity of the test : {test}")
    return  test

def Check_Homogeneity(series:pd.Series)->bool:
    """
    Check if the series is homogenous
    """
    statistic, p_val = stats.shapiro(series)
    logging.debug(f"Statistics : {statistic}")
    logging.debug(f"p value : {p_val}")
    test = p_val > ALPHA_CHECK
    logging.debug(f"Validity of the test : {test}")
    return test

def Check_Size(series:pd.Series)->bool:
    """
    Check if the series is big enough
    """
    len_series = len(series)
    logging.debug(f"size of data : {len_series}")
    logging.debug(f"N_max : {N_max}")
    test = len_series > N_max
    logging.debug(f"Validity of the test : {test}")
    return test

def Check_Proportion_Group(df:pd.DataFrame, columns:list)->bool:
    """
    Check the condition for the proportion
    """
    size_by_group = []
    for index, proportion in enumerate(df):
        logging.debug(f"proportion min = {N_min}")
        if len(proportion) <= 0:
            logging.debug(f"At least 1 value have less than 1 value, first index is : {index}")
            return False
        else:
            size = len(proportion)
            logging.debug(f"size of data : {size}")
            size_by_group.append(size < N_min)
    
    size_by_group = pd.Series(size_by_group)
    logging.debug(f"test proportion of all data : {size_by_group}")
    mean = size_by_group.mean()
    logging.debug(f"mean of all data : {mean}")
    test = size_by_group.mean() >= Percentage_group_prop
    logging.debug(f"Result of the test proportion : {test}")
    if test:
        return False
    else:
        return True
    
def Check_Group(df:pd.DataFrame, columns:list, function:callable)->bool:
    """
    Check if the series is big enough
    """
    logging.debug(f"name of the function : {function}")
    try :
        function in [
            Check_Normality,
            Check_Homogeneity,
            Check_Size
        ]
    except ValueError:
        logging.info("Error : function not found")
        
    try:
        size = len(columns)
        logging.debug(f"number of columns : {size}")
        if size <= 0:
            raise ValueError("Error : df is empty")
    except ValueError as e:
            logging.info(e)

    
    for column in columns:
        df_column = df.loc[:,column]
        if function(df_column) == False:
            return False
    return True

def Check_nb_cols(columns:list, aim:int):
    size = len(columns)
    logging.debug(f"number of columns to test : {size}")
    logging.debug(f"Kind of test by columns : {aim}")
    
    try :
        if size <= 0:
            raise ValueError(f"Error : columns must be a list with at least one element")
    except ValueError as e:
        logging.info(e)
    
    try:
        if aim <= 0:
            raise ValueError(f"Error : You should indicate for 'aim' a positive integer, number of caracteristics you need to compare")
    except ValueError as e:
        logging.info(e)
    
    if aim == 1 or aim == 2:
        try:
            if aim != size:
                raise ValueError(f"Error : columns must be a list with {aim} elements")
        except ValueError as e:
            logging.info(e)
    else:
        try:
            if aim != columns and (aim != 3 or size >= 3):
                raise ValueError(f"Error : aim shall be max 3 or equal to number of columns")
        except ValueError as e:
            logging.info(e)
        
def Check_alpha_value(alpha:float):
    logging.debug(f"Value of alpha : {alpha}")
    try:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Error : alpha must be between 0 and 1, advice 0.05")
    except ValueError as e:
        logging.info(e)
        
def Check_Side(side:str)->bool:
    """
    Check if the side is correct
    """
    logging.debug(f"Value of side : {side}")
    try:
        side in ['Two', 'Left', 'Right']
    except ValueError:
        logging.info("Error : side must be a string in ['Two', 'Left', 'Right']")

def Compare_Mean(df:pd.DataFrame, columns:list, size:int ,alpha:float=0.05, side:str='Two', ref:float=None, dependant:float=True)->float:
    """
    Compare the mean of a Value, Series or a DataFrame
    """
    logging.debug(f"Value of side : {side}")
    try:
        if size == 1:
            logging.debug(f"Using Compare_Mean_to_Ref_Value function")
            return Compare_Mean_to_Ref_Value(df=df, columns=columns, alpha=alpha, side=side, ref=ref)   
        elif size == 2:
            logging.debug(f"Using Compare_two_Means function")
            return Compare_two_Means(df=df, columns=columns, alpha=alpha, side=side, dependant=dependant)
        elif size >= 3:
            logging.debug(f"Using Compare_Mean_to_Ref_Group function")
            return Compare_Mean_to_Ref_Group(df=df, columns=columns, alpha=alpha, side=side)
        else:
            raise ValueError("Error : size must be greater than 0")
    except ValueError as e:
        logging.info(e)

def Compare_Mean_to_Ref_Value(df:pd.DataFrame, columns:list, alpha:float, side:str, ref:float)->bool:
    Check_nb_cols(columns, 1)
    
    try:
        if type(ref) != float:
            raise ValueError("Error : ref must be a float")
    except ValueError as e:
        logging.info(e)
        
    Check_Side(side)
    
    Check_alpha_value(alpha)
    
    logging.info(f"Comparing the values of {columns[0]} to the ref value of {ref}")
        
    if not Check_Size(df[columns[0]]) and not Check_Normality(df[columns[0]]):
        logging.info("Running Non-Parametric Wilcoxon Rank Sum Test")
        differences = df[columns[0]] - ref
        statistic, p = stats.wilcoxon(differences, alternative='two-sided')
        logging.debug(f"Statistics : {statistic}")
        logging.debug(f"p value : {p}")

    else:
        logging.info("Running 1 Sample T-test")
        statistic, p = stats.ttest_1samp(df[columns[0]], ref)
        logging.debug(f"Statistics : {statistic}")
        logging.debug(f"p value : {p}")
    
    test = p < alpha
    logging.debug(f"Validity of the test : {test}")
    return test
        
def Compare_two_Means(df:pd.DataFrame, columns:list, alpha:float, side:str, dependant:bool)->float:
    Check_nb_cols(columns, 2)
        
    Check_alpha_value(alpha)
    
    Check_Side(side)
    logging.info(f"Comparing the values of {columns[0]} to the values of {columns[1]}")
    
    if not dependant:
        if Check_Group(df, columns, Check_Size):
            if Check_Group(df, columns, Check_Homogeneity):
                logging.info("Running Parametric Student T-test")
                statistic, p = stats.ttest_ind(df[columns[0]], df[columns[1]])
                logging.debug(f"Statistics : {statistic}")
                logging.debug(f"p value : {p}")
            else:
                logging.info("Running Parametric Adjusted T-test")
                statistic, p = stats.ttest_ind(df[columns[0]], df[columns[1]], equal_var=False)
                logging.debug(f"Statistics : {statistic}")
                logging.debug(f"p value : {p}") 
                
        elif not Check_Group(df, columns, Check_Normality):
            logging.info("Running Non-Parametric Mann-Whitney U test")
            statistic, p = stats.mannwhitneyu(df[columns[0]], df[columns[1]], alternative='two-sided')
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")
        else:
            if Check_Group(df, columns, Check_Homogeneity):
                logging.info("Running Parametric Student T-test")
                statistic, p = stats.ttest_ind(df[columns[0]], df[columns[1]])
                logging.debug(f"Statistics : {statistic}")
                logging.debug(f"p value : {p}")
            else:
                logging.info("Running Parametric Adjusted T-test")
                statistic, p = stats.ttest_ind(df[columns[0]], df[columns[1]], equal_var=False)
                logging.debug(f"Statistics : {statistic}")
                logging.debug(f"p value : {p}")         
    else:
        if not Check_Group(df, columns, Check_Size) and not Check_Group(df, columns, Check_Normality):
            logging.info("Running Non-Parametric Wilcoxon Rank Sum Test")
            statistic, p = stats.ranksums(df[columns[0]], df[columns[1]])
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
        else:
            logging.info("Running Parametric Paired Sample T-test")
            statistic, p = stats.ttest_rel(df[columns[0]], df[columns[1]])
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  

    test = p < alpha
    logging.debug(f"Validity of the test : {test}")
    return test

def Compare_Mean_to_Ref_Group(df:pd.DataFrame, columns:list, alpha:float, side:str)->float:
    Check_nb_cols(columns, 3)

    Check_alpha_value(alpha)
    
    Check_Side(side)
    
    if Check_Group(df, columns, Check_Size):
        if Check_Group(df, columns, Check_Homogeneity):
            logging.info("Running Parametric One Way ANOVA")
            df = df.loc[:,columns]
            groups = [df[col] for col in df.columns]
            statistic, p = stats.f_oneway(*groups)
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
        else:
            logging.info("Running Non-Parametric Kruskal-Wallis Test")
            df = df.loc[:,columns]
            groups = [df[col] for col in df.columns]
            statistic, p = stats.kruskal(*groups)
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
    else:
        if not Check_Group(df, columns, Check_Normality) and not Check_Group(df, columns, Check_Homogeneity):
            logging.info("Running Non-Parametric Kruskal-Wallis Test")
            df = df.loc[:,columns]
            groups = [df[col] for col in df.columns]
            statistic, p = stats.kruskal(*groups)
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
        else:
            logging.info("Running Parametric One Way ANOVA")
            df = df.loc[:,columns]
            groups = [df[col] for col in df.columns]
            statistic, p = stats.f_oneway(*groups)
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
    
    test = p < alpha
    logging.debug(f"Validity of the test : {test}")
    return test

def Compare_Proportion(df:pd.DataFrame, columns:list, alpha:float, side:str, dependant:bool):
    ## Apporter une transformation en tableau de proportion
    Check_alpha_value(alpha)
    Check_Side(side)
    
    if dependant:
        if Check_Proportion_Group(df, columns):
            logging.info("Running Parametric Chi-Square Test")
            statistic, p, dof, expected = stats.chi2_contingency(df.loc[:,columns])
            logging.debug(f"Statistics : {statistic}")
            logging.debug(f"p value : {p}")  
            logging.debug(f"Degree of Liberties : {dof}")
            logging.debug(f"expected : {expected}")  
        else:
            logging.info("Running Non-Parametric Fisher's Exact Test")
            oddsratio, p = stats.fisher_exact(df.loc[:,columns], alternative='Two')
            logging.debug(f"Statistics : {oddsratio}")
            logging.debug(f"p value : {p}")  
    else:
        logging.info("Running Non-Parametric McNemar's Test")
        statistic, p = mcnemar(df.loc[:,columns], exact=True)
        logging.debug(f"Statistics : {statistic}")
        logging.debug(f"p value : {p}")
        
    test = p < alpha
    logging.debug(f"Validity of the test : {test}")
    return test

# def 