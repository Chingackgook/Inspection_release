from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.EverydayWechat import *
exe = Executor('EverydayWechat', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/EverydayWechat/everyday_wechat/utils/friend_helper.py'
import time
import random
import itchat
from everyday_wechat.utils import config
from everyday_wechat.utils.data_collection import get_bot_info
from everyday_wechat.utils.common import FILEHELPER
'\nProject: EverydayWechat-Github\nCreator: DoubleThunder\nCreate time: 2019-07-12 23:07\nIntroduction: 处理好友消息内容\n'
import time
import random
import itchat

class MockConfig:

    @staticmethod
    def get(key):
        if key == 'wechat_uuid':
            return 'mock_uuid'
        elif key == 'auto_reply_info':
            return {'is_auto_reply': True, 'is_auto_reply_all': False, 'auto_reply_black_uuids': [], 'auto_reply_white_uuids': ['friend_uuid'], 'auto_reply_prefix': 'Auto: ', 'auto_reply_suffix': ''}
        return None
FILEHELPER = 'filehelper'
__all__ = ['handle_friend']

def handle_friend(msg):
    """ 处理好友信息 """
    try:
        if msg['FromUserName'] == MockConfig.get('wechat_uuid') and msg['ToUserName'] != FILEHELPER:
            return
        conf = MockConfig.get('auto_reply_info')
        if not conf.get('is_auto_reply'):
            return
        uuid = FILEHELPER if msg['ToUserName'] == FILEHELPER else msg['FromUserName']
        is_all = conf.get('is_auto_reply_all')
        auto_uuids = conf.get('auto_reply_black_uuids') if is_all else conf.get('auto_reply_white_uuids')
        if is_all and uuid in auto_uuids:
            return
        if not is_all and uuid not in auto_uuids:
            return
        receive_text = msg['text']
        nick_name = FILEHELPER if uuid == FILEHELPER else msg['user']['nickName']
        print('\n{}发来信息：{}'.format(nick_name, receive_text))
        reply_text = exe.run('get_bot_info', message=receive_text, userId=uuid)
        if reply_text:
            time.sleep(random.randint(1, 2))
            prefix = conf.get('auto_reply_prefix', '')
            if prefix:
                reply_text = '{}{}'.format(prefix, reply_text)
            suffix = conf.get('auto_reply_suffix', '')
            if suffix:
                reply_text = '{}{}'.format(reply_text, suffix)
            print('回复{}：{}'.format(nick_name, reply_text))
        else:
            print('自动回复失败\n')
    except Exception as exception:
        print(str(exception))
mock_message = {'FromUserName': 'friend_uuid', 'ToUserName': 'mock_uuid', 'text': 'Hello, how are you?', 'user': {'nickName': 'Friend'}}
handle_friend(mock_message)