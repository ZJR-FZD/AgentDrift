import json
import ast


class BaseAPI:
    def __init__(self, generation_config={}):
        self.sys_prompt = ""
        self.generation_config = generation_config

    def get_system_prompt(self, d=None):
        return self.sys_prompt

    def sanitize_json_str(self, s):
        """Escape raw newlines/tabs inside JSON string values so json.loads can parse."""
        # Non-strings: coerce to str
        if not isinstance(s, str):
            return str(s)

        result = []
        in_string = False
        escape = False
        for ch in s:
            if escape:
                result.append(ch)
                escape = False
            elif ch == '\\':
                result.append(ch)
                escape = True
            elif ch == '"':
                result.append(ch)
                in_string = not in_string
            elif in_string and ch == '\n':
                result.append('\\n')
            elif in_string and ch == '\r':
                result.append('\\r')
            elif in_string and ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        return ''.join(result)

    def is_json(self, s):
        """
        True if ``s`` parses as a non-string JSON value. Supports:
        - standard JSON
        - JSON with raw newlines/tabs inside strings
        - single-quoted Python-ish literals
        - extra outer quotes (case 4: peel recursively)
        """
        try:
            s = self.sanitize_json_str(s)
            data = json.loads(s)
            # Still a string → outer quotes; peel and retry
            if isinstance(data, str):
                return self.is_json(data)
            return True
        except:
            try:
                # Fallback for single quotes etc.; prefer ast.literal_eval over eval
                ast.literal_eval(s)
                return True
            except:
                return False

    def parse_json(self, s):
        """
        Parse a JSON-ish string. Supports:
        - standard JSON
        - JSON with raw newlines/tabs inside strings
        - single-quoted Python-ish literals
        - extra outer quotes (case 4: peel recursively)
        On failure return {}.
        """
        try:
            s = self.sanitize_json_str(s)
            data = json.loads(s)
        except ValueError:
            try:
                # Fallback for single quotes etc.
                data = ast.literal_eval(s)
            except:
                return {}

        # Still a string → peel outer quotes recursively
        if isinstance(data, str):
            return self.parse_json(data)
        return data

    def generate_response(self, messages, tools):
        raise NotImplementedError

if __name__ == '__main__':
    # s = "{"name": "extract_year", "parameters": "{"text": "James Clerk Maxwell published "A Dynamical Theory of the Electromagnetic Field," demonstrating that electric and magnetic fields travel through space as waves in 1863"}"}
    # env = BaseAPI()
    # print(env.is_json(s))
    pass
