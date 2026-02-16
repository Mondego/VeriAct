import os
from utils import read_file_as_str

FORMAT_SYSTEM_MSG = """You are an JML specification generator for Java programs."""

FORMAT_REQUEST_METHOD_SPEC_MSG = """
You are provided with the following Java code:
```
{code}
```
Within the code, there exists a piece of annotation in the form of `// >>>INFILL<<<`, which is acting as a placeholder. Please reply me with appropriate JML specifications (namely `requires` and `ensures` statements) to fill in the place that can fully articulate the behaviors of the corresponding method.
Note that:
1. You SHOULD only reply me with the specifications with nothing else. You CANNOT add other comments to the specifications you generated.
2. You SHOULD specify each `requires` or `ensures` statement in a single line. You CANNOT split the content of an identical statement into multiple lines.
3. There could be cases where there already exists some specifications around the placeholder, BUT you SHOULD always come up with NEW specifications, and CANNOT repeat the specifications already exists within the code given to you.
"""

FORMAT_REQUEST_FIELD_SPEC_MSG = """
You are provided with the following Java code:
```
{code}
```
Within the code, there exists a piece of annotation in the form of `// >>>INFILL<<<`, which is acting as a placeholder. Please reply me with appropriate JML specifications (namely `spec_public` and `invariant` statements) to express the properties of the field below the placeholder.
Note that:
1. You SHOULD only reply me with the specifications with nothing else. You CANNOT add other comments to the specifications you generated.
2. You SHOULD specify each `maintaining` or `decreases` statement in a single line. You CANNOT split the content of an identical statement into multiple lines.
3. There could be cases where there already exists some specifications around the placeholder, BUT you SHOULD always come up with NEW specifications, and CANNOT repeat the specifications already exists within the code given to you.
4. You ONLY needs to consider the field pointed out by the placeholder. You SHOULD NOT consider the properties of other fields within the class.
5. Always consider adding `//@ spec_public` to the fields, especially private ones.
6. If you feel there is nothing to specify about the target field, just say ```//@ invariant true;```.
"""

FORMAT_REQUEST_LOOP_SPEC_MSG = """
You are provided with the following Java code:
```
{code}
```
Within the code, there exists a piece of annotation in the form of `// >>>INFILL<<<`, which is acting as a placeholder. Please reply me with appropriate JML specifications (namely `maintaining` and `decreases` statements) to express the loop invariants of the corresponding loop.
Note that:
1. You SHOULD only reply me with the specifications with nothing else. You CANNOT add other comments to the specifications you generated.
2. You SHOULD specify each `maintaining` or `decreases` statement in a single line. You CANNOT split the content of an identical statement into multiple lines.
3. There could be cases where there already exists some specifications around the placeholder, BUT you SHOULD always come up with NEW specifications, and CANNOT repeat the specifications already exists within the code given to you.
"""

FORMAT_REPLY_SPEC_MSG = """
```
{code}
```
"""

def get_fewshot_context(type:str) -> list:
    if type == "loop":
        return get_fewshot_context_loop()
    elif type == "method":
        return get_fewshot_context_method()
    elif type == "field":
        return get_fewshot_context_field()
    else:
        assert False


def get_fewshot_context_field() -> list:
    return [ {
        "role": "system",
        "content": FORMAT_SYSTEM_MSG
    } ]

def get_fewshot_context_method() -> list:
    res = [ {
        "role": "system",
        "content": FORMAT_SYSTEM_MSG
    } ]
    for i in range(4):
        index = str(i+1)

        current_dir = os.path.dirname(os.path.abspath(__file__))

        code_to_infill = read_file_as_str(current_dir + "/fewshots/{dir}/{num}/task".format(dir="method", num=index))
        res.append({
            "role": "user",
            "content": FORMAT_REQUEST_METHOD_SPEC_MSG.format(code=code_to_infill)
        })
        spec_infilled = read_file_as_str(current_dir + "/fewshots/{dir}/{num}/ans".format(dir="method", num=index))
        res.append({
            "role": "assistant",
            "content": FORMAT_REPLY_SPEC_MSG.format(code=spec_infilled)
        })
    return res


def get_fewshot_context_loop() -> list:
    res = [ {
        "role": "system",
        "content": FORMAT_SYSTEM_MSG
    } ]
    for i in range(4):
        index = str(i+1)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        code_to_infill = read_file_as_str(current_dir + "/fewshots/{dir}/{num}/task".format(dir="loop", num=index))
        res.append({
            "role": "user",
            "content": FORMAT_REQUEST_LOOP_SPEC_MSG.format(code=code_to_infill)
        })
        spec_infilled = read_file_as_str(current_dir + "/fewshots/{dir}/{num}/ans".format(dir="loop", num=index))
        res.append({
            "role": "assistant",
            "content": FORMAT_REPLY_SPEC_MSG.format(code=spec_infilled)
        })
    return res


def get_request_msg(code:str, type:str) -> dict:
    if type == "loop":
        return {
            "role": "user",
            "content": FORMAT_REQUEST_LOOP_SPEC_MSG.format(code=code)
        }
    elif type == "method":
        return {
            "role": "user",
            "content": FORMAT_REQUEST_METHOD_SPEC_MSG.format(code=code)
        }
    elif type == "field":
        return {
            "role": "user",
            "content": FORMAT_REQUEST_FIELD_SPEC_MSG.format(code=code)
        }
    else:
        assert False


# [check] the input context  
def context_to_str(context:list) -> str:
    res = ""
    for msg in context:
        res = res + "{role}:{content}\n".format(role=msg["role"], content=msg["content"])
        res = res + "==============================\n"
    return res