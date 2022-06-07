import boto3
import os

def download_files(bucket, local='./tmp'):

    # s3=boto3.client('s3')
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket) 

    for obj in bucket.objects.all():
        if not os.path.exists(os.path.dirname(local)):
            os.makedirs(os.path.dirname(local))
        bucket.download_file(obj.key, obj.key) # save to same path
        

    

"""
    s3=boto3.client('s3')
    os.makedirs(local)
    list=s3.list_objects(bucket)['Contents']

    for key in list:
        s3.download_file(bucket, key['Key'], key['Key'])
        os.path.join(local, key.get('Key'))
"""
    


