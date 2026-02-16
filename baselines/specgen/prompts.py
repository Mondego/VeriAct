import os
import random
from utils import file2str
from models import create_model_config

FORMAT_INIT_PROMPT = """
Please generate JML specifications for the Java program given below.
```
{src_code}
```
"""

FORMAT_REFINE_PROMPT = """
Your specification got the following error information:

{err_info}

Please generate again.
"""

FORMAT_GENERATION_PROMPT = """
Please generate JML specifications for the Java program given below.
```
{src_code}
```
"""

FORMAT_REFINEMENT_PROMPT = """
The following Java code is instrumented with JML specifications:
```
{code}
```
Verifier failed to verify the specifications given above, with error information as follows:
```
{info}
```
Please refine the specifications, so that it can pass the verification.
"""


class GenerationPrompt:

    def read_oracle_as_msg(self, classname):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        filename_oracle = os.path.join(
            current_dir, "prompts", "oracle", classname, classname + ".java"
        )

        filename_clean = os.path.join(
            current_dir, "prompts", "oracle_clean", classname, classname + ".java"
        )

        # filename_oracle = importlib.resources.files("prompts").joinpath("oracle",classname, classname + ".java").read_text()
        # filename_clean = importlib.resources.files("prompts").joinpath("oracle_clean",classname, classname + ".java").read_text()

        msg_request = {
            "role": "user",
            "content": FORMAT_GENERATION_PROMPT.format(
                src_code=file2str(filename_clean)
            ),
        }
        msg_reply = {
            "role": "assistant",
            "content": "```\n{code}\n```".format(code=file2str(filename_oracle)),
        }
        return [msg_request, msg_reply]

    def randomly_select_prompt(self, oracle_list, num, class_name):
        selected_list = random.sample(oracle_list, num)
        while class_name in selected_list:
            selected_list.remove(class_name)
            selected_list.append(random.choice(oracle_list))
        return selected_list

    def manually_select_prompt(self):
        return ["Neg", "BinarySearch", "BubbleSort", "Calculator", "CopyArray"]

    def create_generation_prompt_config(
        self, input_code, class_name, model, temperature
    ):
        msg_base = {
            "role": "system",
            "content": "You are an JML specification generator for java programs.",
        }
        messages = [msg_base]

        oracle_list = [
            "AddLoop",
            "BinarySearch",
            "BubbleSort",
            "CopyArray",
            "Factorial",
            "FIND_FIRST_IN_SORTED",
            "FindFirstZero",
            "Inverse",
            "LinearSearch",
            "OddEven",
            "Perimeter",
            "SetZero",
            "Smallest",
            "StrPalindrome",
            "TransposeMatrix",
        ]

        prompt_list = self.randomly_select_prompt(oracle_list, 4, class_name)
        # prompt_list = self.manually_select_prompt()

        for prompted_oracle in prompt_list:
            messages.extend(self.read_oracle_as_msg(prompted_oracle))

        msg_request = {
            "role": "user",
            "content": FORMAT_GENERATION_PROMPT.format(src_code=input_code),
        }
        messages.append(msg_request)
        return create_model_config(messages, model, temperature)


class RefinementPrompt:
    def gen_extra_guidance(self, err_info):
        if err_info.find("visibility") != -1:
            return 'To avoid errors related to visibility, you can add "spec_public" specifications to the member variables within the class.'
        elif err_info.find("non-pure") != -1:
            return 'To avoid errors related to non-pure methods, you can add "pure" specifications to the methods that doesn\'t modify any class members.'
        elif err_info.find("NegativeIndex") != -1:
            return 'In case of "PossiblyNegativeIndex", you can add "assume" specifications to ensure that the index is either equal to 0 or greater than 0.'
        elif err_info.find("TooLargeIndex") != -1:
            return 'In case of "PossiblyTooLargeIndex", you can add "assume" specifications to ensure that the index is less than the length of the array.'
        elif (
            err_info.find("ArithmeticOperationRange") != -1
            and err_info.find("negation") != -1
        ):
            return 'To avoid integer overflow in integer negation operation, you can add "assume" specification BEFORE the related code, in order to ensure that the operand is greater than the minimal value that can be expressed.'
        elif err_info.find("overflow") != -1:
            return 'To avoid integer overflow in arithmetic operation, you can add "assume" specification to guarantee that the operation result is within expressable range (smaller than maximum value and bigger than minimum value).'
        elif err_info.find("underflow") != -1:
            return 'To avoid integer underflow in arithmetic operation, you can add "assume" specification to guarantee that the operation result is within expressable range (smaller than maximum value and bigger than minimum value).'
        elif err_info.find("LoopInvariantBeforeLoop") != -1:
            return 'The "LoopInvariantBeforeLoop" error indicates that the loop invariant you stated may be violated before entering the loop. You should consider modifying corresponding "loop_invariant" or "maintaining" specifications.'
        else:
            return ""

    def read_refine_as_msg(self, dirname):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dirpath = os.path.join(current_dir, "prompts", "refine", dirname)
        original_code = file2str(os.path.join(dirpath, "original"))
        err_info = file2str(os.path.join(dirpath, "err_info"))
        refined_code = file2str(os.path.join(dirpath, "refined"))

        # dirpath = importlib.resources.files("prompts").joinpath("refine")
        # original_code = dirpath.joinpath(dirname, "original").read_text()
        # err_info = dirpath.joinpath(dirname, "err_info").read_text()
        # refined_code = dirpath.joinpath(dirname, "refined").read_text()

        msg_request = {
            "role": "user",
            "content": FORMAT_REFINEMENT_PROMPT.format(
                code=original_code, info=err_info
            ),
        }
        msg_reply = {
            "role": "assistant",
            "content": "```\n{code}\n```".format(code=refined_code),
        }
        msg_request["content"] += self.gen_extra_guidance(err_info)
        return [msg_request, msg_reply]

    def extract_err_type(self, err_info):
        prompt_list = []
        keyword_dict = {
            "DivideByZero": "divide_by_zero",
            "visibility": "private_visibility",
            "NegativeIndex": "negative_index",
            "TooLargeIndex": "too_large_index",
            "ArithmeticOperationRange negation": "overflow_negation",
            "overflow sum": "overflow_sum",
            "overflow difference": "overflow_sub",
            "overflow multiply": "overflow_mul",
            "overflow divide": "overflow_div",
            "underflow sum": "underflow_sum",
            "underflow difference": "underflow_sub",
            "underflow multiply": "underflow_mul",
            "underflow divide": "underflow_div",
        }
        for key in keyword_dict:
            keyword_list = key.split(" ")
            flag_all_in = True
            for keyword in keyword_list:
                if err_info.find(keyword) == -1:
                    flag_all_in = False
                    break
            if flag_all_in:
                prompt_list.append(keyword_dict[key])
        return prompt_list

    def create_specialized_patcher_prompt_config(
        self, original_code, err_info, model, temperature
    ):
        msg_base = {
            "role": "system",
            "content": "You are an JML specification generator for java programs.",
        }
        messages = [msg_base]

        prompt_list = self.extract_err_type(err_info)
        for dirname in prompt_list:
            messages.extend(self.read_refine_as_msg(dirname))

        msg_request = {
            "role": "user",
            "content": FORMAT_REFINEMENT_PROMPT.format(
                code=original_code, info=err_info
            ),
        }
        msg_request["content"] += self.gen_extra_guidance(err_info)
        messages.append(msg_request)
        return create_model_config(messages, model, temperature)
