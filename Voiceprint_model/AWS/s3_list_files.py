import logging
import boto3
from botocore.exceptions import ClientError
import os

def print_bucket_contents(bucket_name):
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)

    for file in my_bucket.objects.all():
        print(file.key)


if __name__ == '__main__':
    BUCKET_NAME = 'armgroupproject'
    print_bucket_contents('armgroupproject')

