from braket.aws import AwsSession
import boto3

regions = [
    'us-east-1',
    'us-west-1',
    'us-west-2',
    'eu-north-1',
    'eu-west-2',
]
for region in regions:
    session = AwsSession(boto_session=boto3.Session(region_name=region))
    devices = session.search_devices()
    for device in devices:
        print(f"{device['deviceName']}: {device['deviceArn']}")
