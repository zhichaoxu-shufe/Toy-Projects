import numpy as np

def standard_scaler(input_series):
    reformed_input=[]
    # max_range=np.max(input_lst)-np.min(input_lst)
    for item in input_series:
        if np.isnan(item)==True:
            continue
        else:
            reformed_input.append(item)
    minimum=np.min(reformed_input)
    max_range=np.max(reformed_input)-minimum
    for index in range(len(input_series)):
        if np.isnan(input_series[index])==True:
            continue
        else:
            input_series[index]=((input_series[index]-minimum) / max_range)
        
    return input_series

def standard_scaler_backcount(input_series):
    reformed_input=[]
    # max_range=np.max(input_lst)-np.min(input_lst)
    for item in input_series:
        if isinstance(item, str):
            reformed_input.append(float(item))
            continue
        if np.isnan(item)==True:
            continue
        else:
            reformed_input.append(item)
    minimum=np.min(reformed_input)
    max_range=np.max(reformed_input)-minimum
    for index in range(len(input_series)):
        if np.isnan(input_series[index])==True:
            continue
        else:
            input_series[index]=((input_series[index]-minimum) / max_range)
        
    return input_series

# def categorical_encoder(input_series):


def main():
    a=[np.nan, np.nan, 1, 3, 5, 6]
    b=[19, 18, 24, 20, 21, 26, 27, 23, 18, 22, 32, 17, 25, 16, 29, 22, 31, 28, 30, 19, 33, 13, 49, 18, 18, 43, 54, 19.5, 44, 46, 39, 35, 38, 45, 20, 50, 42, 34, 41, 18, 17]
    print(np.max(b))
    print(np.min(b))
    print(standard_scaler(b))

if __name__=='__main__':
    main()