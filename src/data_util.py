from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
import os



def check_corrupted(df: pd.DataFrame, data_folder_path: str) -> list:
    """
    Double check if there are any corrupted file path
    """
    count_file =0
    count_no_such_file = 0
    drop_i_row = []
    
    for i in tqdm(range(len(df))):
        jpg_image = data_folder_path + "/" + df.image_name.iloc[i]
        if os.path.exists(jpg_image): 
            count_file = count_file +1
        else:
            drop_i_row.append(i)
            count_no_such_file = count_no_such_file + 1
    
    print("Corrupted file path:",count_no_such_file)
    return drop_i_row
    
def find_binary_class_weights(series: Union[pd.Series, list]) -> list:
    """
    Find class weights.
    Scaling by total/2 helps keep the loss to a similar magnitude.
    The sum of the weights of all examples stays the same.
    """
    neg, pos = series.value_counts()
    total_number_img = neg + pos
    weight_for_0 = (1 / neg) * (total_number_img / 2.0)
    weight_for_1 = (1 / pos) * (total_number_img / 2.0)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def define_classes(series: Union[pd.Series, list]) -> [list, int]:
    """
    Define the classes and get number of classes.
    """
    classes_to_predict = sorted(series.unique())
    num_classes = len(classes_to_predict)
    return classes_to_predict,num_classes