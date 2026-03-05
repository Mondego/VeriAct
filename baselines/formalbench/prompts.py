GEN_SYS_MESSAGE = (
    "You are an expert in Java Modeling Language (JML). "
    "You will be provided with Java code snippets."
    "Your task is to generate JML specifications for the given Java code. "
    "The specifications should be written as annotations within the Java code and must be compatible with the OpenJML tool for verification. "
    "Ensure the specifications include detailed preconditions, postconditions, necessary loop invariants, invariants, assertions, and any relevant assumptions. "
)

TWO_SHOT_SYS_MESSAGE = (
    "You are an expert in Java Modeling Language (JML). "
    "You will be provided with Java code snippets. "
    "Your task is to generate JML specifications for the given Java code. "
    "The specifications should be written as annotations within the Java code and must be compatible with the OpenJML tool for verification. "
    "Ensure the specifications include detailed preconditions, postconditions, necessary loop invariants, invariants, assertions, and any relevant assumptions."
    "Please also adhere to the following syntax guidelines for JML:\n"
    "JML text is written in comments that either:\n"
    "a) begin with //@ and end with the end of the line, or\n"
    "b) begin with /*@ and end with */. Lines within such a block comment may have the first non-whitespace characters be a series of @ symbols\n"
)

GEN_QUERY = "Please generate JML specifications for the provided Java code."

_LTM_BREAKDOWN = (
    "Let's break down this problem:\n"
    "1. What are the weakest preconditions for the code? Be sure to include preconditions related to nullness and arithmetic bounds.\n"
    "2. What are the strongest postconditions for the code?\n"
    "3. What necessary specifications are required to prove the above post-conditions? "
    "This includes loop invariants, assertions, assumptions, and ranking functions.\n"
)


def build_messages(
    prompt_type,
    model,
    code,
    example_code1=None,
    example_code2=None,
    example_spec1=None,
    example_spec2=None,
):
    # o1-mini does not support system role; use user role instead
    system_role = "user" if model == "o1-mini" else "system"

    if prompt_type == "zero_shot":
        return [
            {"role": system_role, "content": GEN_SYS_MESSAGE},
            {"role": "user", "content": f"{GEN_QUERY}\n\n### CODE\n{code}\n\n"},
        ]

    elif prompt_type == "zs_cot":
        return [
            {"role": system_role, "content": GEN_SYS_MESSAGE},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{code}\nLet's think step by step !\n\n",
            },
        ]

    elif prompt_type == "two_shot":
        return [
            {"role": system_role, "content": TWO_SHOT_SYS_MESSAGE},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{example_code1}\n\n### SPECIFICATION\n",
            },
            {"role": "assistant", "content": f"{example_spec1}\n\n"},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{example_code2}\n\n### SPECIFICATION\n",
            },
            {"role": "assistant", "content": f"{example_spec2}\n\n"},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{code}\n\n### SPECIFICATION\n",
            },
        ]

    elif prompt_type == "fs_cot":
        return [
            {"role": system_role, "content": GEN_SYS_MESSAGE},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{example_code1}\n\n### RESPONSE\n",
            },
            {"role": "assistant", "content": f"{example_spec1}\n\n"},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{example_code2}\n\n### RESPONSE\n",
            },
            {"role": "assistant", "content": f"{example_spec2}\n\n"},
            {
                "role": "user",
                "content": f"{GEN_QUERY}\n\n### CODE\n{code}\n\nLet's think step by step !\n\n",
            },
        ]

    elif prompt_type == "fs_ltm":
        return [
            {"role": system_role, "content": GEN_SYS_MESSAGE},
            {
                "role": "user",
                "content": (
                    f"{GEN_QUERY}\n\n### CODE\n{example_code1}\n\n"
                    f"{_LTM_BREAKDOWN}"
                    "After answering these questions, let's generate the specifications for the code "
                    "and provide solution after `### SPECIFCIATION'\n"
                ),
            },
            {"role": "assistant", "content": f"### RESPONSE\n{example_spec1}\n\n"},
            {
                "role": "user",
                "content": (
                    f"{GEN_QUERY}\n\n### CODE\n{example_code2}\n\n"
                    f"{_LTM_BREAKDOWN}"
                    "After answering these questions, let's generate the specifications for the code "
                    "and provide solution after `### SPECIFCIATION'\n"
                ),
            },
            {"role": "assistant", "content": f"### RESPONSE\n{example_spec2}\n\n"},
            {
                "role": "user",
                "content": (
                    f"{GEN_QUERY}\n\n### CODE\n{code}\n\n"
                    f"{_LTM_BREAKDOWN}"
                    "After answering these questions, let's generate the specifications for the code "
                    "and provide solution, written between triple backquotes, after `### SPECIFCIATION' \n"
                ),
            },
        ]

    else:
        raise ValueError(f"Invalid prompt type: {prompt_type!r}")
