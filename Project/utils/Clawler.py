from google_images_download import google_images_download

import ssl

from utils.config import Config

ssl._create_default_https_context = ssl._create_unverified_context
config = Config().params


# get iamges from google
def imageCrwaling(keyword, dir):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword,
                 "output_directory": dir,
                 "format": "jpg",
                 "language": "Korean"}
    response.download(arguments)
