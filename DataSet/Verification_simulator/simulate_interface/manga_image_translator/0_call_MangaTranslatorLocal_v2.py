from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.manga_image_translator import *
exe = Executor('manga_image_translator', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/manga-image-translator/manga_translator/__main__.py'
import os
import sys
import asyncio
import logging
from argparse import Namespace
from manga_translator import Config
from manga_translator.args import parser
from manga_translator.args import reparse
from manga_translator.manga_translator import set_main_logger
from manga_translator.manga_translator import load_dictionary
from manga_translator.manga_translator import apply_dictionary
from manga_translator.args import parser
from manga_translator.utils import BASE_PATH
from manga_translator.utils import init_logging
from manga_translator.utils import get_logger
from manga_translator.utils import set_log_level
from manga_translator.utils import natural_sort
from manga_translator.mode.local import MangaTranslatorLocal
from manga_translator.mode.ws import MangaTranslatorWS
from manga_translator.mode.share import MangaShare
import json
import os
import sys
import asyncio
import logging
from argparse import Namespace
from manga_translator import Config
from manga_translator.manga_translator import set_main_logger, load_dictionary, apply_dictionary
from manga_translator.utils import BASE_PATH, init_logging, get_logger, set_log_level, natural_sort
# add
hardcoded_args = {
    'mode': 'local',
    'input': ['/mnt/autor_name/Inspection/Resources/images/test_image.jpg'],
    'dest': '',
    'overwrite': True,
    'pre_dict': None,
    'post_dict': None,
    'verbose': True,
    'attempts': 0,
    'ignore_errors': False,
    'model_dir': None,
    'use_gpu': False,
    'use_gpu_limited': False,
    'font_path': '',
    'kernel_size': 3,
    'context_size': 0,
    'batch_size': 1,
    'batch_concurrent': False,
    'disable_memory_optimization': False,
}
# end add

# origin code:
# hardcoded_args = {'mode': 'local', 'input': [''], 'dest': '', 'overwrite': True, 'pre_dict': None, 'post_dict': None, 'verbose': True}

async def dispatch(args: Namespace):
    args_dict = vars(args)
    logger.info(f'Running in {args.mode} mode')
    if args.mode == 'local':
        if not args.input:
            raise Exception('No input image was supplied. Use -i <image_path>')
        from manga_translator.mode.local import MangaTranslatorLocal
        translator = exe.create_interface_objects(interface_class_name='MangaTranslatorLocal', params=args_dict)
        pre_dict = load_dictionary(args.pre_dict)
        post_dict = load_dictionary(args.post_dict)
        if len(args.input) == 1 and os.path.isfile(args.input[0]):
            dest = os.path.join(FILE_RECORD_PATH, 'final.png')
            args.overwrite = True
            await exe.run('translate_path', path=args.input[0], dest=dest, params=args_dict)
            for textline in translator.textlines:
                textline.text = apply_dictionary(textline.text, pre_dict)
                logger.info(f'Pre-translation dictionary applied: {textline.text}')
            for textline in translator.textlines:
                textline.translation = apply_dictionary(textline.translation, post_dict)
                logger.info(f'Post-translation dictionary applied: {textline.translation}')
        else:
            dest = args.dest
            for path in natural_sort(args.input):
                await exe.run('translate_path', path=path, dest=dest, params=args_dict)
                for textline in translator.textlines:
                    textline.text = apply_dictionary(textline.text, pre_dict)
                    logger.info(f'Pre-translation dictionary applied: {textline.text}')
                for textline in translator.textlines:
                    textline.translation = apply_dictionary(textline.translation, post_dict)
                    logger.info(f'Post-translation dictionary applied: {textline.translation}')
    elif args.mode == 'ws':
        from manga_translator.mode.ws import MangaTranslatorWS
        translator = MangaTranslatorWS(args_dict)
        await translator.listen(args_dict)
    elif args.mode == 'shared':
        from manga_translator.mode.share import MangaShare
        translator = MangaShare(args_dict)
        await translator.listen(args_dict)
    elif args.mode == 'config-help':
        import json
        config = Config.schema()
        print(json.dumps(config, indent=2))

def main():
    args = None
    init_logging()
    try:
        args = Namespace(**hardcoded_args)
        set_log_level(level=logging.DEBUG if args.verbose else logging.INFO)
        logger = get_logger(args.mode)
        set_main_logger(logger)
        if args.mode != 'web':
            logger.debug(args)
        asyncio.run(dispatch(args))
    except KeyboardInterrupt:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except asyncio.CancelledError:
        print('\nTranslation cancelled by user.')
        sys.exit(0)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}', exc_info=e if args and args.verbose else None)
main()