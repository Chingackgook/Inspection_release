# aider 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'aider/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/aider')
# 以上是自动生成的代码，请勿修改


from typing import Any, Dict, List
from abc import ABC, abstractmethod
from aider.coders.base_coder import Coder

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        pass
    def run(self, name: str, **kwargs):
        """
        执行具体智能模块的入口方法，由子类实现，根据name执行对应的方法，并将结果存储在self.result中
        """
        try:
            # 调用 Coder 类及其方法
            if name == 'create':
                coder = Coder.create(**kwargs)
                self.result.interface_return = coder
                self.coder = coder
            elif name == 'clone':
                self.result.interface_return = self.coder.clone(**kwargs)
            elif name == 'get_announcements':
                self.result.interface_return = self.coder.get_announcements()
            elif name == 'show_announcements':
                self.coder.show_announcements()
                self.result.interface_return = None
            elif name == 'add_rel_fname':
                self.coder.add_rel_fname(kwargs['rel_fname'])
                self.result.interface_return = None
            elif name == 'drop_rel_fname':
                success = self.coder.drop_rel_fname(kwargs['fname'])
                self.result.interface_return = success
            elif name == 'abs_root_path':
                self.result.interface_return = self.coder.abs_root_path(kwargs['path'])
            elif name == 'show_pretty':
                self.result.interface_return = self.coder.show_pretty()
            elif name == 'get_abs_fnames_content':
                self.result.interface_return = self.coder.get_abs_fnames_content()
            elif name == 'choose_fence':
                self.coder.choose_fence()
                self.result.interface_return = None
            elif name == 'get_files_content':
                self.result.interface_return = self.coder.get_files_content(kwargs.get('fnames'))
            elif name == 'get_read_only_files_content':
                self.result.interface_return = self.coder.get_read_only_files_content()
            elif name == 'get_cur_message_text':
                self.result.interface_return = self.coder.get_cur_message_text()
            elif name == 'get_ident_mentions':
                self.result.interface_return = self.coder.get_ident_mentions(kwargs['text'])
            elif name == 'get_ident_filename_matches':
                self.result.interface_return = self.coder.get_ident_filename_matches(kwargs['idents'])
            elif name == 'get_repo_map':
                self.result.interface_return = self.coder.get_repo_map(kwargs.get('force_refresh', False))
            elif name == 'get_repo_messages':
                self.result.interface_return = self.coder.get_repo_messages()
            elif name == 'get_readonly_files_messages':
                self.result.interface_return = self.coder.get_readonly_files_messages()
            elif name == 'get_chat_files_messages':
                self.result.interface_return = self.coder.get_chat_files_messages()
            elif name == 'get_images_message':
                self.result.interface_return = self.coder.get_images_message(kwargs['fnames'])
            elif name == 'run_stream':
                self.result.interface_return = self.coder.run_stream(kwargs['user_message'])
            elif name == 'init_before_message':
                self.coder.init_before_message()
                self.result.interface_return = None
            elif name == 'run':
                self.coder.run(with_message=kwargs.get('with_message'), preproc=kwargs.get('preproc', False))
                self.result.interface_return = None
            elif name == 'copy_context':
                self.coder.copy_context()
                self.result.interface_return = None
            elif name == 'get_input':
                self.result.interface_return = self.coder.get_input()
            elif name == 'preproc_user_input':
                self.result.interface_return = self.coder.preproc_user_input(kwargs['inp'])
            elif name == 'run_one':
                self.coder.run_one(kwargs['user_message'], preproc=kwargs.get('preproc', False))
                self.result.interface_return = None
            elif name == 'check_and_open_urls':
                self.result.interface_return = self.coder.check_and_open_urls(kwargs['exc'], kwargs.get('friendly_msg'))
            elif name == 'check_for_urls':
                self.result.interface_return = self.coder.check_for_urls(kwargs['inp'])
            elif name == 'keyboard_interrupt':
                self.coder.keyboard_interrupt()
                self.result.interface_return = None
            elif name == 'summarize_start':
                self.coder.summarize_start()
                self.result.interface_return = None
            elif name == 'summarize_worker':
                self.coder.summarize_worker()
                self.result.interface_return = None
            elif name == 'summarize_end':
                self.coder.summarize_end()
                self.result.interface_return = None
            elif name == 'move_back_cur_messages':
                self.coder.move_back_cur_messages(kwargs['message'])
                self.result.interface_return = None
            elif name == 'get_user_language':
                self.result.interface_return = self.coder.get_user_language()
            elif name == 'get_platform_info':
                self.result.interface_return = self.coder.get_platform_info()
            elif name == 'fmt_system_prompt':
                self.result.interface_return = self.coder.fmt_system_prompt(kwargs['prompt'])
            elif name == 'format_chat_chunks':
                self.result.interface_return = self.coder.format_chat_chunks()
            elif name == 'format_messages':
                self.result.interface_return = self.coder.format_messages()
            elif name == 'warm_cache':
                self.coder.warm_cache(kwargs['chunks'])
                self.result.interface_return = None
            elif name == 'check_tokens':
                self.result.interface_return = self.coder.check_tokens(kwargs['messages'])
            elif name == 'send_message':
                self.result.interface_return = self.coder.send_message(kwargs['inp'])
            elif name == 'show_send_output':
                self.coder.show_send_output(kwargs['completion'])
                self.result.interface_return = None
            elif name == 'show_send_output_stream':
                self.coder.show_send_output_stream(kwargs['completion'])
                self.result.interface_return = None
            elif name == 'live_incremental_response':
                self.coder.live_incremental_response(kwargs['final'])
                self.result.interface_return = None
            elif name == 'render_incremental_response':
                self.result.interface_return = self.coder.render_incremental_response(kwargs['final'])
            elif name == 'remove_reasoning_content':
                self.coder.remove_reasoning_content()
                self.result.interface_return = None
            elif name == 'calculate_and_show_tokens_and_cost':
                self.coder.calculate_and_show_tokens_and_cost(kwargs['messages'], kwargs.get('completion'))
                self.result.interface_return = None
            elif name == 'show_usage_report':
                self.coder.show_usage_report()
                self.result.interface_return = None
            elif name == 'get_multi_response_content_in_progress':
                self.result.interface_return = self.coder.get_multi_response_content_in_progress(kwargs.get('final', False))
            elif name == 'get_rel_fname':
                self.result.interface_return = self.coder.get_rel_fname(kwargs['fname'])
            elif name == 'get_inchat_relative_files':
                self.result.interface_return = self.coder.get_inchat_relative_files()
            elif name == 'is_file_safe':
                self.result.interface_return = self.coder.is_file_safe(kwargs['fname'])
            elif name == 'get_all_relative_files':
                self.result.interface_return = self.coder.get_all_relative_files()
            elif name == 'get_all_abs_files':
                self.result.interface_return = self.coder.get_all_abs_files()
            elif name == 'get_addable_relative_files':
                self.result.interface_return = self.coder.get_addable_relative_files()
            elif name == 'check_for_dirty_commit':
                self.coder.check_for_dirty_commit(kwargs['path'])
                self.result.interface_return = None
            elif name == 'allowed_to_edit':
                self.result.interface_return = self.coder.allowed_to_edit(kwargs['path'])
            elif name == 'check_added_files':
                self.coder.check_added_files()
                self.result.interface_return = None
            elif name == 'prepare_to_edit':
                self.result.interface_return = self.coder.prepare_to_edit(kwargs['edits'])
            elif name == 'apply_updates':
                self.result.interface_return = self.coder.apply_updates()
            elif name == 'parse_partial_args':
                self.result.interface_return = self.coder.parse_partial_args()
            elif name == 'get_context_from_history':
                self.result.interface_return = self.coder.get_context_from_history(kwargs['history'])
            elif name == 'auto_commit':
                self.result.interface_return = self.coder.auto_commit(kwargs['edited'], kwargs.get('context'))
            elif name == 'show_auto_commit_outcome':
                self.coder.show_auto_commit_outcome(kwargs['res'])
                self.result.interface_return = None
            elif name == 'show_undo_hint':
                self.coder.show_undo_hint()
                self.result.interface_return = None
            elif name == 'dirty_commit':
                self.coder.dirty_commit()
                self.result.interface_return = None
            elif name == 'get_edits':
                self.result.interface_return = self.coder.get_edits(kwargs.get('mode'))
            elif name == 'apply_edits':
                self.coder.apply_edits(kwargs['edits'])
                self.result.interface_return = None
            elif name == 'apply_edits_dry_run':
                self.result.interface_return = self.coder.apply_edits_dry_run(kwargs['edits'])
            elif name == 'run_shell_commands':
                self.result.interface_return = self.coder.run_shell_commands()
            elif name == 'handle_shell_commands':
                self.result.interface_return = self.coder.handle_shell_commands(kwargs['commands_str'], kwargs['group'])
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.result.interface_return
            )

        except Exception as e:
            self.result.set_result(
                fuc_name=name,
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('create')
adapter_additional_data['functions'].append('clone')
adapter_additional_data['functions'].append('get_announcements')
adapter_additional_data['functions'].append('show_announcements')
adapter_additional_data['functions'].append('add_rel_fname')
adapter_additional_data['functions'].append('drop_rel_fname')
adapter_additional_data['functions'].append('abs_root_path')
adapter_additional_data['functions'].append('show_pretty')
adapter_additional_data['functions'].append('get_abs_fnames_content')
adapter_additional_data['functions'].append('choose_fence')
adapter_additional_data['functions'].append('get_files_content')
adapter_additional_data['functions'].append('get_read_only_files_content')
adapter_additional_data['functions'].append('get_cur_message_text')
adapter_additional_data['functions'].append('get_ident_mentions')
adapter_additional_data['functions'].append('get_ident_filename_matches')
adapter_additional_data['functions'].append('get_repo_map')
adapter_additional_data['functions'].append('get_repo_messages')
adapter_additional_data['functions'].append('get_readonly_files_messages')
adapter_additional_data['functions'].append('get_chat_files_messages')
adapter_additional_data['functions'].append('get_images_message')
adapter_additional_data['functions'].append('run_stream')
adapter_additional_data['functions'].append('init_before_message')
adapter_additional_data['functions'].append('run')
adapter_additional_data['functions'].append('copy_context')
adapter_additional_data['functions'].append('get_input')
adapter_additional_data['functions'].append('preproc_user_input')
adapter_additional_data['functions'].append('run_one')
adapter_additional_data['functions'].append('check_and_open_urls')
adapter_additional_data['functions'].append('check_for_urls')
adapter_additional_data['functions'].append('keyboard_interrupt')
adapter_additional_data['functions'].append('summarize_start')
adapter_additional_data['functions'].append('summarize_worker')
adapter_additional_data['functions'].append('summarize_end')
adapter_additional_data['functions'].append('move_back_cur_messages')
adapter_additional_data['functions'].append('get_user_language')
adapter_additional_data['functions'].append('get_platform_info')
adapter_additional_data['functions'].append('fmt_system_prompt')
adapter_additional_data['functions'].append('format_chat_chunks')
adapter_additional_data['functions'].append('format_messages')
adapter_additional_data['functions'].append('warm_cache')
adapter_additional_data['functions'].append('check_tokens')
adapter_additional_data['functions'].append('send_message')
adapter_additional_data['functions'].append('show_send_output')
adapter_additional_data['functions'].append('show_send_output_stream')
adapter_additional_data['functions'].append('live_incremental_response')
adapter_additional_data['functions'].append('render_incremental_response')
adapter_additional_data['functions'].append('remove_reasoning_content')
adapter_additional_data['functions'].append('calculate_and_show_tokens_and_cost')
adapter_additional_data['functions'].append('show_usage_report')
adapter_additional_data['functions'].append('get_multi_response_content_in_progress')
adapter_additional_data['functions'].append('get_rel_fname')
adapter_additional_data['functions'].append('get_inchat_relative_files')
adapter_additional_data['functions'].append('is_file_safe')
adapter_additional_data['functions'].append('get_all_relative_files')
adapter_additional_data['functions'].append('get_all_abs_files')
adapter_additional_data['functions'].append('get_addable_relative_files')
adapter_additional_data['functions'].append('check_for_dirty_commit')
adapter_additional_data['functions'].append('allowed_to_edit')
adapter_additional_data['functions'].append('check_added_files')
adapter_additional_data['functions'].append('prepare_to_edit')
adapter_additional_data['functions'].append('apply_updates')
adapter_additional_data['functions'].append('parse_partial_args')
adapter_additional_data['functions'].append('get_context_from_history')
adapter_additional_data['functions'].append('auto_commit')
adapter_additional_data['functions'].append('show_auto_commit_outcome')
adapter_additional_data['functions'].append('show_undo_hint')
adapter_additional_data['functions'].append('dirty_commit')
adapter_additional_data['functions'].append('get_edits')
adapter_additional_data['functions'].append('apply_edits')
adapter_additional_data['functions'].append('apply_edits_dry_run')
adapter_additional_data['functions'].append('run_shell_commands')
adapter_additional_data['functions'].append('handle_shell_commands')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
