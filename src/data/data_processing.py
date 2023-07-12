from matplotlib import pyplot as plt
import pandas as pd
 


def feature_selection(df:pd.DataFrame):
    """This function remove unecessary features

    Args:
        df (pd.DataFrame): Raw dataframe

    Returns:
        pd.DataFrame: Dataframe with the selected attributes
    """
    columns_to_remove = ['first', 'last', 'notes']
    nonexistent_columns = [col for col in columns_to_remove if col not in df.columns]
    try:
        df = df.drop(columns_to_remove, axis=1)
        print("Columns removed successfully.")
    except KeyError:
        print("Error: The following attribute(s) do not exist in the DataFrame:", ", ".join(nonexistent_columns))

    return df

def data_num(df:pd.DataFrame):

    """Convert 'sex' attribute to numerical values using label encoding and
    lang', 'country' attributes to numerical values using one-hot encoding

    Args:
        df (_type_): _description_
     Returns:
        pd.DataFrame: Dataframe with the selected attributes
    """
    columns_to_change = ['sex', 'lang', 'country']

    gender_mapping= {'Male': 1, 'Female': 0}
    try:
        df['sex'] = df['sex'].map(gender_mapping)

        df = pd.get_dummies(df, columns=['lang', 'country'], drop_first=True)
    except KeyError:
        print("Error: The following attribute(s) do not exist in the DataFrame:",", ".join(columns_to_change))
    
    return df


    def pred_data_processing(user_input:dict):
        """User inputs through arguments parser

        Args:
            user_input (dict): dictionary of inputs
        """
        
        attributes_to_remove = ['first', 'last', 'notes']  # List of attributes to remove

        # Remove attributes provided by the user
        for attribute in attributes_to_remove:
            try:
                del user_input[attribute]
                gender_mapping= {'Male': 1, 'Female': 0}
            except KeyError:
                pass
        if user_input.sex =='Male':
            user_input.sex = 1
        elif user_input.sex =="Female":
            user_input.sex ==0
        else:
            print("The gender is not properly defined")
                
                
