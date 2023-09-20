import cv2


from S3_api import S3
s=S3()
s.list_bucket_content('driver01')
print(s.get_object_url('driver01', 'test.jpg'))



print("%%%%%%%%%%%%%%%%%%%%%%%%%%")
s.download_object('obs-f1bd', 'test.jpg', 'D:/test.jpg')
#s.create_bucket('xm-bucket')
#s.upload_object('xm-bucket','kid.jpg','D:/kid2.jpg')
