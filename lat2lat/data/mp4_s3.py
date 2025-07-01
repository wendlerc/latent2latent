import fsspec
import random
import tempfile
import os
import logging
import subprocess
import json
from dotenv import load_dotenv

load_dotenv("/home/developer/workspace/.env2")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MP4Bucket:
    def __init__(self, bucket_name, prefix=''):
        self.bucket_name = bucket_name
        self.prefix = prefix.strip('/')
        self.fs = self._get_fs()

    def _get_fs(self):
        kwargs = {
            'key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
        }
        endpoint = os.getenv('AWS_ENDPOINT_URL_S3')
        if endpoint:
            kwargs['endpoint_url'] = endpoint
        return fsspec.filesystem('s3', **{k:v for k,v in kwargs.items() if v})

    def get_random_keys(self, sample_size=10):
        logger.info(f"🔍 Searching for MP4 files in bucket: {self.bucket_name}, prefix: {self.prefix}")
        
        # build pattern
        if self.prefix:
            pattern = f"{self.bucket_name}/{self.prefix}/**/*.mp4"
        else:
            pattern = f"{self.bucket_name}/**/*.mp4"

        logger.info(f"🔍 Globbing S3 with pattern: {pattern}")
        keys = self.fs.glob(pattern)
        logger.info(f"📁 Found {len(keys)} files with recursive glob")

        # fallback if nothing matched
        if not keys:
            logger.warning("⚠️  No matches for recursive glob—trying non-recursive under prefix")
            if self.prefix:
                pattern2 = f"{self.bucket_name}/{self.prefix}/*.mp4"
                logger.info(f"🔍 Trying pattern: {pattern2}")
                keys = self.fs.glob(pattern2)
                logger.info(f"📁 Found {len(keys)} files with non-recursive glob")

        if not keys:
            logger.error(f"❌ No MP4 files found in s3://{self.bucket_name}/{self.prefix}")
            raise RuntimeError(f"No MP4 files found in s3://{self.bucket_name}/{self.prefix}")
        
        # random sample
        sample_count = min(sample_size, len(keys))
        selected_keys = random.sample(keys, sample_count)
        logger.info(f"🎲 Selected {sample_count} random keys from {len(keys)} available files")
        return selected_keys
    

    def extract_random_crop(self, key, crop_duration=10):
        """
        Extract a random crop using ffmpeg directly with S3 presigned URL.
        Returns path to the clipped file or None on any error.
        """
        logger.info(f"🎬 Starting extraction for key: {key}")
        try:
            # Get file size first
            file_size = self.fs.size(key)
            logger.info(f"📊 File size: {file_size / (1024*1024):.2f} MB")
            
            # 1) Generate presigned URL for S3 access
            logger.info("🔗 Generating presigned URL...")
            presigned_url = self.fs.url(key, expires=3600)  # 1 hour expiry
            logger.info("✅ Generated presigned URL")
            
            # 2) Get video duration using ffprobe
            logger.info("🎞️  Probing video duration with ffprobe...")
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                presigned_url
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"❌ ffprobe failed: {result.stderr}")
                return None
                
            info = json.loads(result.stdout)
            duration = float(info.get('format', {}).get('duration', 0))
            logger.info(f"⏱️  Video duration: {duration:.2f} seconds")
            
            if duration <= crop_duration:
                logger.warning(f"⚠️  Video too short ({duration:.2f}s) for {crop_duration}s crop")
                return None
            
            # 3) Choose random start time
            start = random.uniform(0, duration - crop_duration)
            logger.info(f"✂️  Extracting {crop_duration}s clip starting at {start:.2f}s")
            
            # 4) Extract clip using ffmpeg directly from S3
            out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            out_path = out_file.name
            out_file.close()
            
            logger.info(f"💾 Writing clip to: {out_path}")
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # overwrite output
                '-ss', str(start),  # seek to start time
                '-i', presigned_url,  # input from S3
                '-t', str(crop_duration),  # duration
                '-c:v', 'libx264',  # video codec
                '-c:a', 'aac',  # audio codec
                '-err_detect', 'ignore_err',
                '-fflags', '+igndts',
                '-movflags', '+frag_keyframe+empty_moov',
                '-avoid_negative_ts', 'make_zero',  # handle timing issues
                out_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"❌ ffmpeg failed: {result.stderr}")
                try:
                    os.unlink(out_path)
                except:
                    pass
                return None
            
            logger.info(f"✅ Successfully created crop: {out_path}")
            return out_path

        except Exception as e:
            logger.error(f"❌ Failed to extract crop from {key}: {str(e)}")
            return None



    def sample_video_crops(self, num_samples=3, crop_duration=10):
        logger.info(f"🎯 Starting batch crop extraction: {num_samples} samples, {crop_duration}s each")
        
        keys = self.get_random_keys(num_samples)
        logger.info(f"📝 Selected keys: {keys}")
        
        out = []
        success_count = 0
        
        for i, key in enumerate(keys, 1):
            logger.info(f"▶︎  Processing {i}/{len(keys)}: {key}")
            try:
                f = self.extract_random_crop(key, crop_duration)
                if f:
                    logger.info(f"✅ [{i}/{len(keys)}] Successfully created crop: {f}")
                    out.append(f)
                    success_count += 1
                else:
                    logger.warning(f"⚠️  [{i}/{len(keys)}] Failed to create crop (returned None)")
            except Exception as e:
                logger.error(f"❌ [{i}/{len(keys)}] Exception during crop extraction: {e}")
        
        logger.info(f"🎉 Batch complete: {success_count}/{len(keys)} successful crops")
        return out


if __name__ == "__main__":
    logger.info("🚀 Starting MP4 crop extraction script")
    bucket = MP4Bucket('cod-yt-playlist')
    crops = bucket.sample_video_crops(num_samples=3, crop_duration=10)
    logger.info(f"🏁 Script complete. Generated {len(crops)} crops: {crops}")
