import pandas as pd
import numpy as np

#turn target_col to 0/1
def transform_target_var_binary(df,y_col_name,positive_state_str):
    df[y_col_name] = np.where(df[y_col_name] == positive_state_str, 1, 0)
    return df

def k_target_encode_getGroups(df,k_groups):
    #divide the data into k equal (as equal as possible) sized datasets
    n_rows = len(df)
    
    smallest_group_size = int(n_rows/k_groups)
    remaining_rows = n_rows - (smallest_group_size * k_groups)
    
    sample_from_df = df.copy(deep=True)
    #print(sample_from_df.shape)
    
    groups = list()
    
    for i in range(k_groups):
        selection = df.sample(n=smallest_group_size,replace=False)
        #selection = selection.drop("Unnamed: 0",axis=1)
        groups.append(selection)
        
        for inv,val in selection.iterrows():
            sample_from_df = sample_from_df[sample_from_df["Unnamed: 0"] != val["Unnamed: 0"]]
            
    for i in range(remaining_rows):
        groups[i] = np.append(groups[i],np.random.choice(sample_from_df.index.values,1,replace=False))
        sample_from_df.drop(groups[i][-1])
        
    #print(groups[0].shape)
    
    return groups

def k_target_encode_getMeanByValueByGroup(group,unique_values,encode_col,target_col):
    vallvl = {}
    valCountByGroup = {}
    for uv in unique_values:
        locdf = group.copy(deep=True)
        locdf = locdf[locdf[encode_col] == uv]
        counter = 0
        for index,row in locdf.iterrows():
            if row[target_col] == 1:
                counter = counter + 1
        vallvl[uv] = (counter,len(locdf))


    return vallvl

def k_target_encode_getOverallMeanByGroup(group,target_col):
    count = 0
    total = 0
    for ind,row in group.iterrows():
        count = count + row[target_col]
        total = total + 1
    result = count / total
    return result   

def k_target_encode(df, encode_col, target_col, k_groups, positive_state_str, m_overall_mean_weight=2):
    #check to make sure that target column is binary
    if df[target_col].nunique() != 2:
        print('non-binary target column don\'t work')
        return None
    
    #transform target column to 0/1 with 1 being the positive state
    df = transform_target_var_binary(df,'diet','low fat')

    #check how many categories there are in the encode_col
    n_unique = df[encode_col].nunique()
    unique_vals = list(df[encode_col].unique())
    
    #divide the data into k equal (as equal as possible) sized datasets
    groups = k_target_encode_getGroups(df,k_groups)
    group_lengths = list()
    group_means = list()
    for group in groups:
        group_lengths.append(len(group))
        group_means.append(k_target_encode_getOverallMeanByGroup(group,target_col))
    
    #get the option means (the means of the unique values) of each group gotten
    option_means = list()
    group_counts = list()
    for group in groups:
        unique_values = list(group[encode_col].unique())
        this_group_counts = k_target_encode_getMeanByValueByGroup(group,unique_values,encode_col,target_col)
        for uv in unique_vals:
            if (uv not in this_group_counts.keys()):
                this_group_counts[uv] = 0
        group_counts.append(this_group_counts)
    
    #get the weighted means for each unique value for each group
    #the formula is N(n_rows_used_for_option_mean) * OM(average of option means for uv from all OTHER groups)
    #    + M(parameter) * FM(overall mean from all OTHER groups)
    #    ALL THAT DIVIDED BY N + M
    weighted_means_all = list()
    for i,group in enumerate(groups):
        #calculate for one group
        counts_for_n_calc = {}
        for v in unique_vals:
            counts_for_n_calc[v] = [0,0]
        
        for j,g in enumerate(groups):
            if i != j:
                for v in unique_vals:
                    counts_for_n_calc[v][0] = counts_for_n_calc[v][0] + group_counts[j][v][0]
                    counts_for_n_calc[v][1] = counts_for_n_calc[v][1] + group_counts[j][v][1] 
        
        option_means = {}
        for v in unique_vals:
            option_means[v] = counts_for_n_calc[v][0]/counts_for_n_calc[v][1]
        
        weighted_means = {}
        for v in unique_vals:
            weighted_means[v] = option_means[v] * counts_for_n_calc[v][1]
            weighted_means[v] = weighted_means[v] + (m_overall_mean_weight * group_means[i])
            weighted_means[v] = weighted_means[v] / (1 + m_overall_mean_weight)
            
        weighted_means_all.append(weighted_means)
    
    encoded_df = pd.DataFrame()
    for i,group in enumerate(groups):
        for index,row in group.iterrows():
            this_row = row
            this_row[encode_col] = weighted_means_all[i][row[encode_col]]
            encoded_df = pd.concat([encoded_df,this_row.to_frame().T])

    return encoded_df