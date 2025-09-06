from abc import ABC, abstractmethod
from typing import Any

class ExecutionResult:
    # Template class for execution results, used to encapsulate the outcome after invoking an interface.
    def __init__(self):
        self.fuc_name = ''  # Name of the executed function/method
        self.is_success = False  # Whether execution was successful
        self.fail_reason = ''  # Reason for execution failure
        self.is_file = False  # Whether the result is a file
        self.file_path = ''  # File path if the result is a file
        self.except_data = None  # Expected result, in case the valuable outcome is not returned via the method but through other means
        self.interface_return = None  # Original interface return value

    def set_result(
        self,
        fuc_name: str,
        is_success: bool,
        fail_reason: str,
        is_file: bool,
        file_path: str,
        except_data: Any,
        interface_return: Any,
    ):
        self.fuc_name = fuc_name
        self.is_success = is_success
        self.fail_reason = fail_reason
        self.is_file = is_file
        self.file_path = file_path
        self.interface_return = interface_return
        self.except_data = except_data


class BaseAdapter(ABC):

    def __init__(self):
        # Result returned by the model after execution
        self.result: ExecutionResult = ExecutionResult()

    @abstractmethod
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        """
        Method to initialize interface objects. This should be implemented in CustomAdapter,
        which creates interface objects based on `interface_class_name` and `kwargs`.
        The result should be stored in `self.result`. If the interface class is not required, pass.
        """
        pass
        """
        Example template:
        try:
            if interface_class_name == 'Class1Name':
                # Create interface object
                self.class1_obj = Class1Name(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'Class2Name':
                self.class2_obj = Class2Name(**kwargs)
                self.result.interface_return = self.class2_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.class1_obj = Class1Name(**kwargs)
                self.result.interface_return = self.class1_obj

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")
        """
        pass

    @abstractmethod
    def run(self, dispatch_key: str, **kwargs):
        """
        Entry point for execution. Should be implemented by CustomAdapter.
        Executes corresponding methods/functions based on `dispatch_key`,
        and stores the result in `self.result`.
        Use `if dispatch_key == 'xxx'` to determine which method to execute.
        """
        pass
        """
        Example template:
        try:
            if dispatch_key == 'xxx':
                # Call function interface
                self.result.interface_return = xxx(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'class1Name_class_methodName':
                # Call class method from class1(if it exists)
                self.result.interface_return = Class1Name.class_methodName(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'class1Name_static_methodName':
                # Call static method from class1(if it exists)
                self.result.interface_return = Class1Name.static_methodName(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'class1Name_methodName':
                # Call method from class1
                self.result.interface_return = self.class1_obj.methodName(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'class2Name_methodName':
                # Call method from class2
                self.result.interface_return = self.class2_obj.methodName(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")
        """

"""
This is a base class template.
Please implement a CustomAdapter that inherits from BaseAdapter,
and implement the methods `create_interface_objects` and `run` based on the interface documentation provided.

In the `run` method, `dispatch_key` corresponds to a method name in the interface documentation,
and `kwargs` contains the arguments for that method.

The return value from the original interface must be stored in `self.result.interface_return`.
"""

# Example usage (Only for reference):
# adapter = CustomAdapter()
# adapter.create_interface_objects(**kwargs)
# adapter.run(dispatch_key='interface_name', **kwargs)
# print(adapter.result)
