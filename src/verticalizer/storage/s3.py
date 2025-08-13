import os
import logging
from typing import Optional
import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

def _client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT or None,
        aws_access_key_id=S3_ACCESS_KEY or None,
        aws_secret_access_key=S3_SECRET_KEY or None,
        region_name=S3_REGION,
        config=Config(s3={"addressing_style": "path"})
    )

def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream"):
    if not S3_BUCKET:
        logger.warning("[S3] No bucket configured; skipping put for %s", key)
        return
    c = _client()
    c.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    logger.info(f"[S3] Put s3://{S3_BUCKET}/{key}")

def get_bytes(key: str) -> Optional[bytes]:
    if not S3_BUCKET:
        logger.warning("[S3] No bucket configured; skipping get for %s", key)
        return None
    c = _client()
    try:
        r = c.get_object(Bucket=S3_BUCKET, Key=key)
        return r["Body"].read()
    except Exception as e:
        logger.warning(f"[S3] Get failed for {key}: {e}")
        return None
