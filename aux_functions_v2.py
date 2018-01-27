import h2o

def normalize_data(df, to_scale=[],to_keep=[],to_h2o = True):
    from sklearn import preprocessing
    import pandas as pd
    import h2o
    
    if len(to_scale) == 0:
        to_scale = df.select_dtypes(exclude=['object','bool']).columns#df.loc[:,df.dtypes != object].column
    
    if len(to_keep) == 0:
        to_keep = df.select_dtypes(include=['object','bool' ]).columns#df.loc[:,df.dtypes == object].column
        
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_data = min_max_scaler.fit_transform(df[to_scale])
    
    scaled_data = pd.DataFrame(scaled_data,columns=to_scale)
    scaled_data.index.name = 'scale_index'
    normalized_data = df[to_keep]
    normalized_data.index.name = 'normalized_data_index'
    if to_h2o:
        final_data = h2o.H2OFrame(pd.merge(scaled_data, normalized_data, left_index=True, right_index=True))
        return final_data
    final_data = pd.merge(scaled_data, normalized_data, left_index=True, right_index=True)
    return final_data

def drop_columns(df,columns_to_drop=[]):
    for column in columns_to_drop:
        df = df.drop(column)
    return df

def column_to_factors(df,columns_to_convert=[]):
    if isinstance(df, type(h2o.H2OFrame())):
        df = df.as_data_frame()
    if len(columns_to_convert) == 0:
        columns_to_convert = df.select_dtypes(include=['object','bool','int']).columns#df.loc[:,(df.dtypes == object)|(df.dtypes == bool)].columns
    df = h2o.H2OFrame(df)
    for column in columns_to_convert:
        df[column] = df[column].asfactor()
    #df = df.as_data_frame(use_pandas=True)
    return df

    return df
def add_interactions(df):
    from itertools import combinations
    from sklearn.preprocessing import PolynomialFeatures
    import pandas as pd

    df_factor_values_only = df.select_dtypes(include=['object','bool' ])
    df_numeric_values_only = df.select_dtypes(exclude=['object','bool'])

    combos = list(combinations(list(df_numeric_values_only.columns),2))    
    colnames = list(df_numeric_values_only.columns) + ['_'.join(x) for x in combos]

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    df = poly.fit_transform(df_numeric_values_only)

    df = pd.DataFrame(df)
    df.columns = colnames

    noint_indicies = [i for i,x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis = 1)

    df = pd.merge(df, df_factor_values_only, left_index=True, right_index=True)
    
    return df

def clean_data(df, to_factor=[], to_drop=[], to_scale=[], normalize=False, validation_set = False):
    import h2o
    
    dropped_data = drop_columns(df = df)
    data = column_to_factors(df = dropped_data)
    data = add_interactions(data)
    
    data = h2o.H2OFrame(data)
    training_columns = data.col_names[:-1]
    response_column = 'exited'

    if normalize:
        data = data.as_data_frame(use_pandas=True)
        data = normalize_data(df=data,to_scale=to_factor,to_keep=to_factor)

    if validation_set:
        train, valid, test = data.split_frame([0.65,0.15], seed=1234)
        return data,train, valid, test,training_columns,response_column  
    else:
        train, test = data.split_frame([0.70], seed=1234)
        return data, train, test, training_columns, response_column   

def upsample(data_frame,minority_to_majority_ratio,to_keep=[]):
    from sklearn.utils import resample
    import pandas as pd
    df_majority = data_frame[data_frame["exited"].asnumeric() == 0,:].as_data_frame(use_pandas=True)
    df_minority = data_frame[data_frame["exited"].asnumeric() == 1,:].as_data_frame(use_pandas=True)

    df_minority_upsampled = resample(df_minority, 
                                     replace=True,                      # sample with replacement
                                     n_samples=int(round(int(df_majority['exited'].count())*minority_to_majority_ratio)),# to match majority class
                                     random_state=123)                  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    #data_frame = h2o.H2OFrame(df_upsampled)
    final_frame = column_to_factors(df = df_upsampled) 
    return final_frame


