import requests
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

def check_existing_file(file_path):
    '''Check if a file already exists. If it does, ask if we want to overwrite it.'''
    if os.path.isfile(file_path):
        while True:
            response = input(f"File {os.path.basename(file_path)} already exists. Do you want to overwrite it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return True
    
    
def check_existing_folder(folder_path):
    '''Check if a folder already exists. If it doesn't, ask if we want to create it.'''
    if os.path.exists(folder_path) == False :
        while True:
            response = input(f"{os.path.basename(folder_path)} doesn't exists. Do you want to create it? (y/n): ")
            if response.lower() == 'y':
                return True
            elif response.lower() == 'n':
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return False

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''import filenames from bucket_folder_url in raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames :
        input_file = os.path.join(bucket_folder_url,filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f'downloading {input_file} as {os.path.basename(output_file)}')
            response = requests.get(object_url)
            if response.status_code == 200:
                # Process the response content as needed
                content = response.text
                text_file = open(output_file, "wb")
                text_file.write(content.encode('utf-8'))
                text_file.close()
            else:
                print(f'Error accessing the object {input_file}:', response.status_code)


                
def main(raw_data_relative_path="data/raw", 
        filenames = ["admission.csv"],
        bucket_folder_url= "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv"          
        ):
    """ Upload data from AWS s3 in ./
    """
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    # logger = logging.getLogger(__name__)
    # logger.info('making raw data set')
    # file_path = os.path.join(raw_data_relative_path, filenames[0])
    # df = pd.read_csv(file_path, sep=",")
    # df = df.drop('date', axis=1)
    # print(df.head())

    # target = df['silica_concentrate']
    # feats = df.drop(['silica_concentrate'], axis = 1)

    # X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state = 42)

    # for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
    #     output_filepath = os.path.join("data/processed_data", f'{filename}.csv')
    #     file.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()