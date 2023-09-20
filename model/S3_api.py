import boto.s3.connection
from functools import wraps

ACCESS_KEY = ''
SECRET_KEY = ''
HOST = 'obs.cn-north-4.myhuaweicloud.com'

def is_exist_bucket(func):
    @wraps(func)
    def is_exist_bucket_decorator(self, *args, **kwargs):
        bucket_name = args[0]
        if bucket_name not in [bucket.name for bucket in
                               self.conn.get_all_buckets()]:
            self.conn.create_bucket(bucket_name)
            # print(func(*args, **kwargs))
        return func(self,*args, **kwargs)

    return is_exist_bucket_decorator

class S3:

    def __init__(self):
        self.url_validity = 3600
        self.conn = boto.connect_s3(
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            host=HOST,
            is_secure=False,
            calling_format=boto.s3.connection.OrdinaryCallingFormat(),
        )
        self.conn.auth_region_name = 'cn-north-4'

    @is_exist_bucket
    def upload_object(self, bucket_name, object_name, object_path):  # 上传对象到指定的存储桶
        try:
            bucket = self.conn.get_bucket(bucket_name)
            key = bucket.new_key(object_name)
            key.set_contents_from_filename(object_path)
            return True
        except Exception as e:
            print("upload {} error:{}".format(object_name, e))
            return False

    def get_object_url(self, bucket_name, object_name): # 获取对象的URL
        try:
            bucket = self.conn.get_bucket(bucket_name)
            plans_key = bucket.get_key(object_name)
            plans_url = plans_key.generate_url(self.url_validity,
                                               query_auth=True,
                                               force_http=False)
            return plans_url
        except Exception as e:
            print("get {} error:{}".format(object_name, e))
            return False

    def download_object(self, bucket_name, object_names, data_path,
                        compressed_name=None):   # 下载对象
        try:
            bucket = self.conn.get_bucket(bucket_name)
            key = bucket.get_key(object_names)
            key.get_contents_to_filename(data_path)
            return True
        except Exception as e:
            print("download error: %s" % e)
            return False

    def list_bucket_content(self, bucket_name):   # 列出存储桶中的对象列表
        try:
            bucket = self.conn.get_bucket(bucket_name)
            for key in bucket.list():
                print("{name}\t{size}\t{modified}".format(
                    name=key.name,
                    size=key.size,
                    modified=key.last_modified,
                ))
        except Exception as e:
            print("get {} error:{}".format(bucket_name, e))
            return False

    def create_bucket(self, bucket_name):   # 创建桶
        try:
            self.conn.create_bucket(bucket_name,headers=None,location='cn-north-4', policy=None)
            return True
        except Exception as e:
            print("get {} error:{}".format(bucket_name, e))
            return False

    def delete_bucket(self, bucket_name):   # 删除存储桶
        try:
            self.conn.delete_bucket(bucket_name)
        except Exception as e:
            print("get {} error:{}".format(bucket_name, e))
            return False

    def delete_object(self,bucket_name,object_name):   # 删除对象
        try:
            bucket = self.conn.get_bucket(bucket_name)
            bucket.delete_key(object_name)
            return True
        except Exception as e:
            print("get {} error:{}".format(bucket_name, e))
            return False
