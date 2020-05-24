import boto3
import cred

s3 = boto3.client('s3', region_name='ap-southeast-1', aws_access_key_id=cred.ACCESS_KEY, aws_secret_access_key=cred.SECRET_KEY)
bucket_name= 'yolo-model'


s3.upload_file('yolo_model/yolo.pth', bucket_name, 'yolo.pth')
