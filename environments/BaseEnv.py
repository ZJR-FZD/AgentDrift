import json
import os
import inspect

class BaseEnv:
    def __init__(self):
        self.tool_list = []
        class_name = self.__class__.__name__

        # Resolve schema JSON next to the subclass source file.
        dirpath = os.path.dirname(inspect.getfile(self.__class__))
        json_path = os.path.join(dirpath, f"{class_name}.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, list):
            # Format 1: Direct list of tool descriptions
            self.tool_descs = data
        elif isinstance(data, dict):
            if 'tools' in data:
                # Format 2: Dictionary with 'tools' key containing list
                self.tool_descs = data['tools']
            else:
                # Format 3: Single tool description or other structure
                # If it has 'name' and 'parameters', treat as single tool
                if 'name' in data and 'parameters' in data:
                    self.tool_descs = [data]
                else:
                    # Unknown format, try to extract tools from dictionary values
                    self.tool_descs = []
                    for key, value in data.items():
                        if isinstance(value, dict) and 'name' in value and 'parameters' in value:
                            self.tool_descs.append(value)
                    if not self.tool_descs:
                        raise ValueError(f"Invalid JSON format in {json_path}")

        for tool_desc in self.tool_descs:
            self.tool_list.append(tool_desc['name'])

    def call_tool(self, tool_name, arguments):
        if not hasattr(self, tool_name):
            return {'success': False, 'message': f"Invalid tool name {tool_name}."}
        func = getattr(self, tool_name)
        tool_desc = self.get_tool_descs([tool_name])[0]
        if 'required' in tool_desc['parameters']:
            for required in tool_desc['parameters']['required']:
                if required not in arguments:
                    return {'success': False, 'message': f"Missing required parameter {required} for tool {tool_name}."}
        for key in list(arguments.keys()):
            if key not in tool_desc["parameters"]["properties"]:
                del arguments[key]
        try:
            for key in arguments:
                t = tool_desc["parameters"]["properties"][key]["type"]
                if t == "integer":
                    arguments[key] = int(arguments[key])
                elif t == "number":
                    arguments[key] = float(arguments[key])
                elif t == "boolean":
                    arguments[key] = bool(arguments[key])
                elif t == "object" and not isinstance(arguments[key], dict):
                    arguments[key] = json.loads(arguments[key])
                elif t == "array" and not isinstance(arguments[key], list):
                    arguments[key] = json.loads(arguments[key])
        except:
            return {'success': False, 'message': f"Parameter {key} should have type {t}."}
        return func(**arguments)

    def get_tool_descs(self, tool_names):
        tool_descs = []
        for tool_name in tool_names:
            found = False
            for td in self.tool_descs:
                if td['name'] == tool_name:
                    tool_descs.append(td)
                    found = True
                    break
            if not found:
                raise Exception(f"Tool {tool_name} not found in {self.__class__.__name__}")
        return tool_descs

    def has_tool(self, tool_name):
        return tool_name in self.tool_list