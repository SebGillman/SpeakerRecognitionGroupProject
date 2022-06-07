import boto3
import os

def download_files(bucket, local='./tmp'):

    s3=boto3.client('s3')
    os.makedirs(local)
    list=s3.list_objects(bucket)['Contents']

    for key in list:
        s3.download_file(bucket, key['Key'], key['Key'])
        os.path.join(local, key.get('Key'))