# gpt_researcher 
from curses import raw
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gpt_researcher/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/gpt_researcher')
# 以上是自动生成的代码，请勿修改


from typing import Any, Dict, List
from abc import ABC, abstractmethod
from gpt_researcher import GPTResearcher

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.GPTResearcher = None

    def create_interface_objects(self, **kwargs):
        # 初始化 GPTResearcher 类
        self.GPTResearcher = GPTResearcher(**kwargs)
        self.result.set_result(
            fuc_name='create_interface_objects',
            is_success=True,
            fail_reason='',
            is_file=False,
            file_path='',
            except_data=None,
            interface_return=None
        )

    def run(self, name: str, **kwargs):
        try:
            if name == 'conduct_research':
                context = self.GPTResearcher.conduct_research(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=context,
                    interface_return=context
                )
            elif name == 'write_report':
                report = self.GPTResearcher.write_report(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=report,
                    interface_return=report
                )
            elif name == 'write_report_conclusion':
                conclusion = self.GPTResearcher.write_report_conclusion(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=conclusion,
                    interface_return=conclusion
                )
            elif name == 'write_introduction':
                intro = self.GPTResearcher.write_introduction()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=intro,
                    interface_return=intro
                )
            elif name == 'quick_search':
                results = self.GPTResearcher.quick_search(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=results,
                    interface_return=results
                )
            elif name == 'get_subtopics':
                subtopics = self.GPTResearcher.get_subtopics()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=subtopics,
                    interface_return=subtopics
                )
            elif name == 'get_draft_section_titles':
                draft_section_titles = self.GPTResearcher.get_draft_section_titles(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=draft_section_titles,
                    interface_return=draft_section_titles
                )
            elif name == 'get_similar_written_contents_by_draft_section_titles':
                similar_contents = self.GPTResearcher.get_similar_written_contents_by_draft_section_titles(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=similar_contents,
                    interface_return=similar_contents
                )
            elif name == 'get_research_images':
                images = self.GPTResearcher.get_research_images(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=images,
                    interface_return=images
                )
            elif name == 'add_research_images':
                self.GPTResearcher.add_research_images(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'get_research_sources':
                sources = self.GPTResearcher.get_research_sources()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=sources,
                    interface_return=sources
                )
            elif name == 'add_research_sources':
                self.GPTResearcher.add_research_sources(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'add_references':
                updated_markdown = self.GPTResearcher.add_references(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=updated_markdown,
                    interface_return=updated_markdown
                )
            elif name == 'extract_headers':
                headers = self.GPTResearcher.extract_headers(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=headers,
                    interface_return=headers
                )
            elif name == 'extract_sections':
                sections = self.GPTResearcher.extract_sections(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=sections,
                    interface_return=sections
                )
            elif name == 'table_of_contents':
                toc = self.GPTResearcher.table_of_contents(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=toc,
                    interface_return=toc
                )
            elif name == 'get_source_urls':
                urls = self.GPTResearcher.get_source_urls()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=urls,
                    interface_return=urls
                )
            elif name == 'get_research_context':
                context = self.GPTResearcher.get_research_context()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=context,
                    interface_return=context
                )
            elif name == 'get_costs':
                costs = self.GPTResearcher.get_costs()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=costs,
                    interface_return=costs
                )
            elif name == 'set_verbose':
                self.GPTResearcher.set_verbose(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'add_costs':
                self.GPTResearcher.add_costs(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == '_log_event':
                self.GPTResearcher._log_event(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == '_handle_deep_research':
                context = self.GPTResearcher._handle_deep_research(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=context,
                    interface_return=context
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='Method not found',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
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
adapter_additional_data['functions'].append('conduct_research')
adapter_additional_data['functions'].append('write_report')
adapter_additional_data['functions'].append('write_report_conclusion')
adapter_additional_data['functions'].append('write_introduction')
adapter_additional_data['functions'].append('quick_search')
adapter_additional_data['functions'].append('get_subtopics')
adapter_additional_data['functions'].append('get_draft_section_titles')
adapter_additional_data['functions'].append('get_similar_written_contents_by_draft_section_titles')
adapter_additional_data['functions'].append('get_research_images')
adapter_additional_data['functions'].append('add_research_images')
adapter_additional_data['functions'].append('get_research_sources')
adapter_additional_data['functions'].append('add_research_sources')
adapter_additional_data['functions'].append('add_references')
adapter_additional_data['functions'].append('extract_headers')
adapter_additional_data['functions'].append('extract_sections')
adapter_additional_data['functions'].append('table_of_contents')
adapter_additional_data['functions'].append('get_source_urls')
adapter_additional_data['functions'].append('get_research_context')
adapter_additional_data['functions'].append('get_costs')
adapter_additional_data['functions'].append('set_verbose')
adapter_additional_data['functions'].append('add_costs')
adapter_additional_data['functions'].append('_log_event')
adapter_additional_data['functions'].append('_handle_deep_research')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
