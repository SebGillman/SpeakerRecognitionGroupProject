import boto3
import os

def download_files(bucket, local='./tmp'):
    # s3=boto3.client('s3')
    s3_client = boto3.client('s3')

    for obj in bucket.objects.all():
        source = obj.key
        destination = os.path.join(local, source)
        if not os.path.exists(os.path.dirname(destination)):
            os.makedirs(os.path.dirname(destination))
        s3_client.download_file(bucket, source, destination)




"""
from cloudpathlib import CloudPath
def download_files(bucket, local='./tmp'):

    if not os.path.exists(local):
        os.makedirs(local)

    path = "s3://"+bucket
    cp = CloudPath(path)
    cp.download_to(local)
"""

"""
import boto3
import os

def download_files(bucket, local='./tmp'):
    # s3=boto3.client('s3')
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket) 
    os.makedirs(local)
    cur_path = os.getcwd()

    for obj in bucket.objects.all():
        file = bucket.download_file(obj.key, obj.key) # save to same path

        os.path.join(cur_path, local, file)
"""



"""
    s3=boto3.client('s3')
    os.makedirs(local)
    list=s3.list_objects(bucket)['Contents']

    for key in list:
        s3.download_file(bucket, key['Key'], key['Key'])
        os.path.join(local, key.get('Key'))
"""
    


