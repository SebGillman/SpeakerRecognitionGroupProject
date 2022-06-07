import boto3
import os

def download_files(bucket, local='./tmp'):
    # s3=boto3.client('s3')

    def get_all_s3_objects(s3, **base_kwargs):
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = s3.list_objects_v2(**list_kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')
            
    s3_client = boto3.client('s3')
    
    all_s3_objects_gen = get_all_s3_objects(s3_client, Bucket=bucket)

    for obj in all_s3_objects_gen:
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
    


