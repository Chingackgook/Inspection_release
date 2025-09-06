from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.txtai import *
exe = Executor('txtai', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/txtai/examples/images.py'
import glob
import os
import sys
import streamlit as st
from PIL import Image
from txtai.embeddings import Embeddings
'\nBuilds a similarity index for a directory of images\n\nRequires streamlit to be installed.\n  pip install streamlit\n'

class Application:
    """
    Main application
    """

    def __init__(self, directory):
        """
        Creates a new application.

        Args:
            directory: directory of images
        """
        self.embeddings = exe.create_interface_objects(interface_class_name='Embeddings', config={'method': 'sentence-transformers', 'path': 'clip-ViT-B-32'})
        self.embeddings.index(self.images(directory))

    def build(self, directory):
        """
        Builds an image embeddings index.

        Args:
            directory: directory with images

        Returns:
            Embeddings index
        """
        embeddings = exe.create_interface_objects(interface_class_name='Embeddings', config={'method': 'sentence-transformers', 'path': 'clip-ViT-B-32'})
        embeddings.index(self.images(directory))
        embeddings.config['path'] = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
        embeddings.model = embeddings.loadvectors()
        return embeddings

    def images(self, directory):
        """
        Generator that loops over each image in a directory.

        Args:
            directory: directory with images
        """
        for path in glob.glob(directory + '/*jpg') + glob.glob(directory + '/*png'):
            yield (path, Image.open(path), None)

    def run(self):
        """
        Runs a search application.
        """
        print('Image search application initialized.')
        print('This application shows how images and text can be embedded into the same space to support similarity search.')
        print('Enter a search query to find similar images.')
        query = 'example search query'
        (index, _) = exe.run('search', query=query, limit=1)[0]
        print(f"Search result for query '{query}': {index}")

def create(directory):
    """
    Creates a Streamlit application.

    Args:
        directory: directory of images to index

    Returns:
        Application
    """
    return Application(directory)
directory_path = RESOURCES_PATH + 'images/test_images_floder'
app = create(directory_path)
app.run()