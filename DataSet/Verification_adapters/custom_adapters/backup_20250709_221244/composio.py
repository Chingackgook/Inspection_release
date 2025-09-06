# composio 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'composio/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/composio')
# 以上是自动生成的代码，请勿修改


from typing import Any, Dict
from composio.tools.toolset import ComposioToolSet

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.toolset = None  # To hold the ComposioToolSet instance

    def create_interface_objects(self, **kwargs):
        """
        Initialize the ComposioToolSet with the provided API key and base URL.
        """
        try:
            self.toolset = ComposioToolSet(**kwargs)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.toolset
            )
        except Exception as e:
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

    def run(self, name: str, **kwargs):
        """
        Execute the specified method of the ComposioToolSet based on the name parameter.
        """
        try:
            if name == 'check_connected_account':
                action = kwargs.get('action')
                entity_id = kwargs.get('entity_id')
                self.toolset.check_connected_account(action=action, entity_id=entity_id)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'execute_action':
                action = kwargs.get('action')
                params = kwargs.get('params', {})
                metadata = kwargs.get('metadata')
                entity_id = kwargs.get('entity_id')
                connected_account_id = kwargs.get('connected_account_id')
                text = kwargs.get('text')
                response = self.toolset.execute_action(
                    action=action,
                    params=params,
                    metadata=metadata,
                    entity_id=entity_id,
                    connected_account_id=connected_account_id,
                    text=text
                )
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=response
                )
            elif name == 'execute_request':
                endpoint = kwargs.get('endpoint')
                method = kwargs.get('method')
                body = kwargs.get('body')
                parameters = kwargs.get('parameters')
                connection_id = kwargs.get('connection_id')
                app = kwargs.get('app')
                response = self.toolset.execute_request(
                    endpoint=endpoint,
                    method=method,
                    body=body,
                    parameters=parameters,
                    connection_id=connection_id,
                    app=app
                )
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=response
                )
            elif name == 'validate_tools':
                apps = kwargs.get('apps')
                actions = kwargs.get('actions')
                tags = kwargs.get('tags')
                self.toolset.validate_tools(apps=apps, actions=actions, tags=tags)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'get_action_schemas':
                apps = kwargs.get('apps')
                actions = kwargs.get('actions')
                tags = kwargs.get('tags')
                check_connected_accounts = kwargs.get('check_connected_accounts', True)
                _populate_requested = kwargs.get('_populate_requested', False)
                schemas = self.toolset.get_action_schemas(
                    apps=apps,
                    actions=actions,
                    tags=tags,
                    check_connected_accounts=check_connected_accounts,
                    _populate_requested=_populate_requested
                )
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=schemas,
                    interface_return=schemas
                )
            elif name == 'create_trigger_listener':
                timeout = kwargs.get('timeout', 15.0)
                listener = self.toolset.create_trigger_listener(timeout=timeout)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=listener,
                    interface_return=listener
                )
            else:
                raise ValueError(f"Unknown method name: {name}")
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
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('check_connected_account')
adapter_addtional_data['functions'].append('execute_action')
adapter_addtional_data['functions'].append('execute_request')
adapter_addtional_data['functions'].append('validate_tools')
adapter_addtional_data['functions'].append('get_action_schemas')
adapter_addtional_data['functions'].append('create_trigger_listener')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
