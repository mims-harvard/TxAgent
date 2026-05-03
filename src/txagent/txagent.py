import gradio as gr
import os
import sys
import json
import gc
import logging
import numpy as np
import torch
from vllm import LLM, SamplingParams
from jinja2 import Template
from typing import List
import types
from tooluniverse import ToolUniverse
from gradio import ChatMessage
from .toolrag import ToolRAGModel

from .utils import NoRepeatSentenceProcessor, ReasoningTraceChecker, tool_result_format

logger = logging.getLogger(__name__)


class TxAgent:
    def __init__(self, model_name,
                 rag_model_name,
                 tool_files_dict=None,  # None leads to the default tool files in ToolUniverse
                 enable_finish=True,
                 enable_rag=True,
                 enable_summary=False,
                 init_rag_num=0,
                 step_rag_num=10,
                 summary_mode='step',
                 summary_skip_last_k=0,
                 summary_context_length=None,
                 force_finish=True,
                 avoid_repeat=True,
                 seed=None,
                 enable_checker=False,
                 enable_chat=False,
                 additional_default_tools=None,
                 ):
        self.model_name = model_name
        self.tokenizer = None
        self.terminators = None
        self.rag_model_name = rag_model_name
        self.tool_files_dict = tool_files_dict
        self.model = None
        self.rag_model = ToolRAGModel(rag_model_name)
        self.tooluniverse = None
        # self.tool_desc = None
        self.prompt_multi_step = "You are a helpful assistant that will solve problems through detailed, step-by-step reasoning and actions based on your reasoning. Typically, your actions will use the provided functions. You have access to the following functions."
        self.self_prompt = "Strictly follow the instruction."
        self.chat_prompt = "You are helpful assistant to chat with the user."
        self.enable_finish = enable_finish
        self.enable_rag = enable_rag
        self.enable_summary = enable_summary
        self.summary_mode = summary_mode
        self.summary_skip_last_k = summary_skip_last_k
        self.summary_context_length = summary_context_length
        self.init_rag_num = init_rag_num
        self.step_rag_num = step_rag_num
        self.force_finish = force_finish
        self.avoid_repeat = avoid_repeat
        self.seed = seed
        self.enable_checker = enable_checker
        self.additional_default_tools = additional_default_tools
        self.print_self_values()

    def init_model(self):
        self.load_models()
        self.load_tooluniverse()
        self.load_tool_desc_embedding()

    def print_self_values(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    def load_models(self, model_name=None):
        if model_name is not None:
            if model_name == self.model_name:
                return f"The model {model_name} is already loaded."
            self.model_name = model_name

        self.model = LLM(model=self.model_name)
        self.chat_template = Template(self.model.get_tokenizer().chat_template)
        self.tokenizer = self.model.get_tokenizer()

        return f"Model {model_name} loaded successfully."

    def load_tooluniverse(self):
        self.tooluniverse = ToolUniverse(tool_files=self.tool_files_dict)
        self.tooluniverse.load_tools()
        special_tools = self.tooluniverse.prepare_tool_prompts(
            self.tooluniverse.tool_category_dicts["special_tools"])
        self.special_tools_name = [tool['name'] for tool in special_tools]

    def load_tool_desc_embedding(self):
        self.rag_model.load_tool_desc_embedding(self.tooluniverse)

    def rag_infer(self, query, top_k=5):
        return self.rag_model.rag_infer(query, top_k)

    def initialize_tools_prompt(self, call_agent, call_agent_level, message):
        picked_tools_prompt = []
        picked_tools_prompt = self.add_special_tools(
            picked_tools_prompt, call_agent=call_agent)
        if call_agent:
            call_agent_level += 1
            if call_agent_level >= 2:
                call_agent = False

        if not call_agent:
            picked_tools_prompt += self.tool_RAG(
                message=message, rag_num=self.init_rag_num)
        return picked_tools_prompt, call_agent_level

    def initialize_conversation(self, message, conversation=None, history=None):
        if conversation is None:
            conversation = []

        conversation = self.set_system_prompt(
            conversation, self.prompt_multi_step)
        if history is not None:
            if len(history) == 0:
                conversation = []
                print("clear conversation successfully")
            else:
                for i in range(len(history)):
                    if history[i]['role'] == 'user':
                        if i-1 >= 0 and history[i-1]['role'] == 'assistant':
                            conversation.append(
                                {"role": "assistant", "content": history[i-1]['content']})
                        conversation.append(
                            {"role": "user", "content": history[i]['content']})
                    if i == len(history)-1 and history[i]['role'] == 'assistant':
                        conversation.append(
                            {"role": "assistant", "content": history[i]['content']})

        conversation.append({"role": "user", "content": message})

        return conversation

    def tool_RAG(self, message=None,
                 picked_tool_names=None,
                 existing_tools_prompt=[],
                 rag_num=5,
                 return_call_result=False):
        extra_factor = 30  # Factor to retrieve more than rag_num
        if picked_tool_names is None:
            assert picked_tool_names is not None or message is not None
            picked_tool_names = self.rag_infer(
                message, top_k=rag_num*extra_factor)

        picked_tool_names_no_special = []
        for tool in picked_tool_names:
            if tool not in self.special_tools_name:
                picked_tool_names_no_special.append(tool)
        picked_tool_names_no_special = picked_tool_names_no_special[:rag_num]
        picked_tool_names = picked_tool_names_no_special[:rag_num]

        picked_tools = self.tooluniverse.get_tool_by_name(picked_tool_names)
        picked_tools_prompt = self.tooluniverse.prepare_tool_prompts(
            picked_tools)
        if return_call_result:
            return picked_tools_prompt, picked_tool_names
        return picked_tools_prompt

    def add_special_tools(self, tools, call_agent=False):
        if self.enable_finish:
            tools.append(self.tooluniverse.get_one_tool_by_one_name(
                'Finish', return_prompt=True))
            print("Finish tool is added")
        if call_agent:
            tools.append(self.tooluniverse.get_one_tool_by_one_name(
                'CallAgent', return_prompt=True))
            print("CallAgent tool is added")
        else:
            if self.enable_rag:
                tools.append(self.tooluniverse.get_one_tool_by_one_name(
                    'Tool_RAG', return_prompt=True))
                print("Tool_RAG tool is added")

            if self.additional_default_tools is not None:
                for each_tool_name in self.additional_default_tools:
                    tool_prompt = self.tooluniverse.get_one_tool_by_one_name(
                        each_tool_name, return_prompt=True)
                    if tool_prompt is not None:
                        print(f"{each_tool_name} tool is added")
                        tools.append(tool_prompt)
        return tools

    def add_finish_tools(self, tools):
        tools.append(self.tooluniverse.get_one_tool_by_one_name(
            'Finish', return_prompt=True))
        print("Finish tool is added")
        return tools

    def set_system_prompt(self, conversation, sys_prompt):
        if len(conversation) == 0:
            conversation.append(
                {"role": "system", "content": sys_prompt})
        else:
            conversation[0] = {"role": "system", "content": sys_prompt}
        return conversation

    def run_function_call(self, fcall_str,
                          return_message=False,
                          existing_tools_prompt=None,
                          message_for_call_agent=None,
                          call_agent=False,
                          call_agent_level=None,
                          temperature=None):

        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=return_message, verbose=False)
        call_results = []
        special_tool_call = ''
        if function_call_json is not None:
            if isinstance(function_call_json, list):
                for i in range(len(function_call_json)):
                    print("\033[94mTool Call:\033[0m", function_call_json[i])
                    if function_call_json[i]["name"] == 'Finish':
                        special_tool_call = 'Finish'
                        break
                    elif function_call_json[i]["name"] == 'Tool_RAG':
                        new_tools_prompt, call_result = self.tool_RAG(
                            message=message,
                            existing_tools_prompt=existing_tools_prompt,
                            rag_num=self.step_rag_num,
                            return_call_result=True)
                        existing_tools_prompt += new_tools_prompt
                    elif function_call_json[i]["name"] == 'CallAgent':
                        if call_agent_level < 2 and call_agent:
                            solution_plan = function_call_json[i]['arguments']['solution']
                            full_message = (
                                message_for_call_agent +
                                "\nYou must follow the following plan to answer the question: " +
                                str(solution_plan)
                            )
                            call_result = self.run_multistep_agent(
                                full_message, temperature=temperature,
                                max_new_tokens=1024, max_token=99999,
                                call_agent=False, call_agent_level=call_agent_level)
                            call_result = call_result.split(
                                '[FinalAnswer]')[-1].strip()
                        else:
                            call_result = "Error: The CallAgent has been disabled. Please proceed with your reasoning process to solve this question."
                    else:
                        call_result = self.tooluniverse.run_one_function(
                            function_call_json[i])

                    call_id = self.tooluniverse.call_id_gen()
                    function_call_json[i]["call_id"] = call_id
                    print("\033[94mTool Call Result:\033[0m", call_result)
                    call_results.append({
                        "role": "tool",
                        "content": json.dumps({"content": call_result, "call_id": call_id})
                    })
        else:
            call_results.append({
                "role": "tool",
                "content": json.dumps({"content": "Not a valid function call, please check the function call format."})
            })

        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results

        # Yield the final result.
        return revised_messages, existing_tools_prompt, special_tool_call

    def run_function_call_stream(self, fcall_str,
                                 return_message=False,
                                 existing_tools_prompt=None,
                                 message_for_call_agent=None,
                                 call_agent=False,
                                 call_agent_level=None,
                                 temperature=None,
                                 return_gradio_history=True):

        function_call_json, message = self.tooluniverse.extract_function_call_json(
            fcall_str, return_message=return_message, verbose=False)
        call_results = []
        special_tool_call = ''
        if return_gradio_history:
            gradio_history = []
        if function_call_json is not None:
            if isinstance(function_call_json, list):
                for i in range(len(function_call_json)):
                    if function_call_json[i]["name"] == 'Finish':
                        special_tool_call = 'Finish'
                        break
                    elif function_call_json[i]["name"] == 'Tool_RAG':
                        new_tools_prompt, call_result = self.tool_RAG(
                            message=message,
                            existing_tools_prompt=existing_tools_prompt,
                            rag_num=self.step_rag_num,
                            return_call_result=True)
                        existing_tools_prompt += new_tools_prompt
                    elif function_call_json[i]["name"] == 'DirectResponse':
                        call_result = function_call_json[i]['arguments']['respose']
                        special_tool_call = 'DirectResponse'
                    elif function_call_json[i]["name"] == 'RequireClarification':
                        call_result = function_call_json[i]['arguments']['unclear_question']
                        special_tool_call = 'RequireClarification'
                    elif function_call_json[i]["name"] == 'CallAgent':
                        if call_agent_level < 2 and call_agent:
                            solution_plan = function_call_json[i]['arguments']['solution']
                            full_message = (
                                message_for_call_agent +
                                "\nYou must follow the following plan to answer the question: " +
                                str(solution_plan)
                            )
                            sub_agent_task = "Sub TxAgent plan: " + \
                                str(solution_plan)
                            # When streaming, yield responses as they arrive.
                            call_result = yield from self.run_gradio_chat(
                                full_message, history=[], temperature=temperature,
                                max_new_tokens=1024, max_token=99999,
                                call_agent=False, call_agent_level=call_agent_level,
                                conversation=None,
                                sub_agent_task=sub_agent_task)

                            call_result = call_result.split(
                                '[FinalAnswer]')[-1]
                        else:
                            call_result = "Error: The CallAgent has been disabled. Please proceed with your reasoning process to solve this question."
                    else:
                        call_result = self.tooluniverse.run_one_function(
                            function_call_json[i])

                    call_id = self.tooluniverse.call_id_gen()
                    function_call_json[i]["call_id"] = call_id
                    call_results.append({
                        "role": "tool",
                        "content": json.dumps({"content": call_result, "call_id": call_id})
                    })
                    if return_gradio_history and function_call_json[i]["name"] != 'Finish':
                        if function_call_json[i]["name"] == 'Tool_RAG':
                            gradio_history.append(ChatMessage(role="assistant", content=str(call_result), metadata={
                                                  "title": "🧰 "+function_call_json[i]['name'], "log": str(function_call_json[i]['arguments'])}))

                        else:
                            gradio_history.append(ChatMessage(role="assistant", content=str(call_result), metadata={
                                                  "title": "⚒️ "+function_call_json[i]['name'], "log": str(function_call_json[i]['arguments'])}))
        else:
            call_results.append({
                "role": "tool",
                "content": json.dumps({"content": "Not a valid function call, please check the function call format."})
            })

        revised_messages = [{
            "role": "assistant",
            "content": message.strip(),
            "tool_calls": json.dumps(function_call_json)
        }] + call_results

        # Yield the final result.
        if return_gradio_history:
            return revised_messages, existing_tools_prompt, special_tool_call, gradio_history
        else:
            return revised_messages, existing_tools_prompt, special_tool_call

    def get_answer_based_on_unfinished_reasoning(self, conversation, temperature, max_new_tokens, max_token, outputs=None, return_full_thought=False):
        if conversation[-1]['role'] == 'assisant':
            conversation.append(
                {'role': 'tool', 'content': 'Errors happen during the function call, please come up with the final answer with the current information.'})
        finish_tools_prompt = self.add_finish_tools([])
        final_thought = 'Since I cannot continue reasoning, I will provide the final answer based on the current information and general knowledge.\n\n[FinalAnswer]'
        last_outputs_str = self.llm_infer(messages=conversation,
                                          temperature=temperature,
                                          tools=finish_tools_prompt,
                                          output_begin_string='Since I cannot continue reasoning, I will provide the final answer based on the current information and general knowledge.\n\n[FinalAnswer]',
                                          skip_special_tokens=True,
                                          max_new_tokens=max_new_tokens, max_token=max_token)
        print(last_outputs_str)
        if return_full_thought:
            return final_thought+last_outputs_str
        return last_outputs_str

    def run_multistep_agent(self, message: str,
                            temperature: float,
                            max_new_tokens: int,
                            max_token: int,
                            max_round: int = 20,
                            call_agent=False,
                            call_agent_level=0) -> str:
        """
        Generate a streaming response using the llama3-8b model.
        Args:
            message (str): The input message.
            temperature (float): The temperature for generating the response.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            str: The generated response.
        """
        print("\033[1;32;40mstart\033[0m")
        picked_tools_prompt, call_agent_level = self.initialize_tools_prompt(
            call_agent, call_agent_level, message)
        conversation = self.initialize_conversation(message)

        outputs = []
        last_outputs = []
        next_round = True
        function_call_messages = []
        current_round = 0
        token_overflow = False
        enable_summary = False
        last_status = {}

        if self.enable_checker:
            checker = ReasoningTraceChecker(message, conversation)
        try:
            while next_round and current_round < max_round:
                current_round += 1
                if len(outputs) > 0:
                    function_call_messages, picked_tools_prompt, special_tool_call = self.run_function_call(
                        last_outputs, return_message=True,
                        existing_tools_prompt=picked_tools_prompt,
                        message_for_call_agent=message,
                        call_agent=call_agent,
                        call_agent_level=call_agent_level,
                        temperature=temperature)

                    if special_tool_call == 'Finish':
                        next_round = False
                        conversation.extend(function_call_messages)
                        if isinstance(function_call_messages[0]['content'], types.GeneratorType):
                            function_call_messages[0]['content'] = next(
                                function_call_messages[0]['content'])
                        return function_call_messages[0]['content'].split('[FinalAnswer]')[-1]

                    if (self.enable_summary or token_overflow) and not call_agent:
                        if token_overflow:
                            print("token_overflow, using summary")
                        enable_summary = True
                    last_status = self.function_result_summary(
                        conversation, status=last_status, enable_summary=enable_summary)

                    if function_call_messages is not None:
                        conversation.extend(function_call_messages)
                        outputs.append(tool_result_format(
                            function_call_messages))
                    else:
                        next_round = False
                        conversation.extend(
                            [{"role": "assistant", "content": ''.join(last_outputs)}])
                        return ''.join(last_outputs).replace("</s>", "")
                if self.enable_checker:
                    good_status, wrong_info = checker.check_conversation()
                    if not good_status:
                        next_round = False
                        print(
                            "Internal error in reasoning: " + wrong_info)
                        break
                last_outputs = []
                outputs.append("### TxAgent:\n")
                last_outputs_str, token_overflow = self.llm_infer(messages=conversation,
                                                                  temperature=temperature,
                                                                  tools=picked_tools_prompt,
                                                                  skip_special_tokens=False,
                                                                  max_new_tokens=max_new_tokens, max_token=max_token,
                                                                  check_token_status=True)
                if last_outputs_str is None:
                    next_round = False
                    print(
                        "The number of tokens exceeds the maximum limit.")
                else:
                    last_outputs.append(last_outputs_str)
            if max_round == current_round:
                print("The number of rounds exceeds the maximum limit!")
            if self.force_finish:
                return self.get_answer_based_on_unfinished_reasoning(conversation, temperature, max_new_tokens, max_token)
            else:
                return None

        except Exception as e:
            print(f"Error: {e}")
            if self.force_finish:
                return self.get_answer_based_on_unfinished_reasoning(conversation, temperature, max_new_tokens, max_token)
            else:
                return None

    def build_logits_processor(self, messages, llm):
        # Use the tokenizer from the LLM instance.
        tokenizer = llm.get_tokenizer()
        if self.avoid_repeat and len(messages) > 2:
            assistant_messages = []
            for i in range(1, len(messages) + 1):
                if messages[-i]['role'] == 'assistant':
                    assistant_messages.append(messages[-i]['content'])
                    if len(assistant_messages) == 2:
                        break
            forbidden_ids = [tokenizer.encode(
                msg, add_special_tokens=False) for msg in assistant_messages]
            return [NoRepeatSentenceProcessor(forbidden_ids, 5)]
        else:
            return None

    def llm_infer(self, messages, temperature=0.1, tools=None,
                  output_begin_string=None, max_new_tokens=2048,
                  max_token=None, skip_special_tokens=True,
                  model=None, tokenizer=None, terminators=None, seed=None, check_token_status=False):

        if model is None:
            model = self.model

        logits_processor = self.build_logits_processor(messages, model)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            logits_processors=logits_processor,
            seed=seed if seed is not None else self.seed,
        )

        prompt = self.chat_template.render(
            messages=messages, tools=tools, add_generation_prompt=True)
        if output_begin_string is not None:
            prompt += output_begin_string

        if check_token_status and max_token is not None:
            token_overflow = False
            num_input_tokens = len(self.tokenizer.encode(
                prompt, return_tensors="pt")[0])
            if max_token is not None:
                if num_input_tokens > max_token:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Number of input tokens before inference:",
                          num_input_tokens)
                    logger.info(
                        "The number of tokens exceeds the maximum limit!!!!")
                    token_overflow = True
                    return None, token_overflow
        output = model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        output = output[0].outputs[0].text
        print("\033[92m" + output + "\033[0m")
        if check_token_status and max_token is not None:
            return output, token_overflow

        return output

    def run_self_agent(self, message: str,
                       temperature: float,
                       max_new_tokens: int,
                       max_token: int) -> str:

        print("\033[1;32;40mstart self agent\033[0m")
        conversation = []
        conversation = self.set_system_prompt(conversation, self.self_prompt)
        conversation.append({"role": "user", "content": message})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_chat_agent(self, message: str,
                       temperature: float,
                       max_new_tokens: int,
                       max_token: int) -> str:

        print("\033[1;32;40mstart chat agent\033[0m")
        conversation = []
        conversation = self.set_system_prompt(conversation, self.chat_prompt)
        conversation.append({"role": "user", "content": message})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_format_agent(self, message: str,
                         answer: str,
                         temperature: float,
                         max_new_tokens: int,
                         max_token: int) -> str:

        print("\033[1;32;40mstart format agent\033[0m")
        if '[FinalAnswer]' in answer:
            possible_final_answer = answer.split("[FinalAnswer]")[-1]
        elif "\n\n" in answer:
            possible_final_answer = answer.split("\n\n")[-1]
        else:
            possible_final_answer = answer.strip()
        if len(possible_final_answer) == 1:
            choice = possible_final_answer[0]
            if choice in ['A', 'B', 'C', 'D', 'E']:
                return choice
        elif len(possible_final_answer) > 1:
            if possible_final_answer[1] == ':':
                choice = possible_final_answer[0]
                if choice in ['A', 'B', 'C', 'D', 'E']:
                    print("choice", choice)
                    return choice

        conversation = []
        format_prompt = f"You are helpful assistant to transform the answer of agent to the final answer of 'A', 'B', 'C', 'D'."
        conversation = self.set_system_prompt(conversation, format_prompt)
        conversation.append({"role": "user", "content": message +
                            "\nThe final answer of agent:" + answer + "\n The answer is (must be a letter):"})
        return self.llm_infer(messages=conversation,
                              temperature=temperature,
                              tools=None,
                              max_new_tokens=max_new_tokens, max_token=max_token)

    def run_summary_agent(self, thought_calls: str,
                          function_response: str,
                          temperature: float,
                          max_new_tokens: int,
                          max_token: int) -> str:
        print("\033[1;32;40mSummarized Tool Result:\033[0m")
        generate_tool_result_summary_training_prompt = """Thought and function calls: 
{thought_calls}

Function calls' responses:
\"\"\"
{function_response}
\"\"\"

Based on the Thought and function calls, and the function calls' responses, you need to generate a summary of the function calls' responses that fulfills the requirements of the thought. The summary MUST BE ONE sentence and include all necessary information.

Directly respond with the summarized sentence of the function calls' responses only. 

Generate **one summarized sentence** about "function calls' responses" with necessary information, and respond with a string:
            """.format(thought_calls=thought_calls, function_response=function_response)
        conversation = []
        conversation.append(
            {"role": "user", "content": generate_tool_result_summary_training_prompt})
        output = self.llm_infer(messages=conversation,
                                temperature=temperature,
                                tools=None,
                                max_new_tokens=max_new_tokens, max_token=max_token)

        if '[' in output:
            output = output.split('[')[0]
        return output

    def function_result_summary(self, input_list, status, enable_summary):
        """
        Processes the input list, extracting information from sequences of 'user', 'tool', 'assistant' roles.
        Supports 'length' and 'step' modes, and skips the last 'k' groups.

        Parameters:
            input_list (list): A list of dictionaries containing role and other information.
            summary_skip_last_k (int): Number of groups to skip from the end. Defaults to 0.
            summary_context_length (int): The context length threshold for the 'length' mode.
            last_processed_index (tuple or int): The last processed index.

        Returns:
            list: A list of extracted information from valid sequences.
        """
        if 'tool_call_step' not in status:
            status['tool_call_step'] = 0

        for idx in range(len(input_list)):
            pos_id = len(input_list)-idx-1
            if input_list[pos_id]['role'] == 'assistant':
                if 'tool_calls' in input_list[pos_id]:
                    if 'Tool_RAG' in str(input_list[pos_id]['tool_calls']):
                        status['tool_call_step'] += 1
                break

        if 'step' in status:
            status['step'] += 1
        else:
            status['step'] = 0

        if not enable_summary:
            return status

        if 'summarized_index' not in status:
            status['summarized_index'] = 0

        if 'summarized_step' not in status:
            status['summarized_step'] = 0

        if 'previous_length' not in status:
            status['previous_length'] = 0

        if 'history' not in status:
            status['history'] = []

        function_response = ''
        idx = 0
        current_summarized_index = status['summarized_index']

        status['history'].append(self.summary_mode == 'step' and status['summarized_step']
                                 < status['step']-status['tool_call_step']-self.summary_skip_last_k)

        idx = current_summarized_index
        while idx < len(input_list):
            if (self.summary_mode == 'step' and status['summarized_step'] < status['step']-status['tool_call_step']-self.summary_skip_last_k) or (self.summary_mode == 'length' and status['previous_length'] > self.summary_context_length):

                if input_list[idx]['role'] == 'assistant':
                    if 'Tool_RAG' in str(input_list[idx]['tool_calls']):
                        this_thought_calls = None
                    else:
                        if len(function_response) != 0:
                            print("internal summary")
                            status['summarized_step'] += 1
                            result_summary = self.run_summary_agent(
                                thought_calls=this_thought_calls,
                                function_response=function_response,
                                temperature=0.1,
                                max_new_tokens=1024,
                                max_token=99999
                            )

                            input_list.insert(
                                last_call_idx+1, {'role': 'tool', 'content': result_summary})
                            status['summarized_index'] = last_call_idx + 2
                            idx += 1

                        last_call_idx = idx
                        this_thought_calls = input_list[idx]['content'] + \
                            input_list[idx]['tool_calls']
                        function_response = ''

                elif input_list[idx]['role'] == 'tool' and this_thought_calls is not None:
                    function_response += input_list[idx]['content']
                    del input_list[idx]
                    idx -= 1

            else:
                break
            idx += 1

        if len(function_response) != 0:
            status['summarized_step'] += 1
            result_summary = self.run_summary_agent(
                thought_calls=this_thought_calls,
                function_response=function_response,
                temperature=0.1,
                max_new_tokens=1024,
                max_token=99999
            )

            tool_calls = json.loads(input_list[last_call_idx]['tool_calls'])
            for tool_call in tool_calls:
                del tool_call['call_id']
            input_list[last_call_idx]['tool_calls'] = json.dumps(tool_calls)
            input_list.insert(
                last_call_idx+1, {'role': 'tool', 'content': result_summary})
            status['summarized_index'] = last_call_idx + 2

        return status

    # Following are Gradio related functions

    # General update method that accepts any new arguments through kwargs
    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Return the updated attributes
        updated_attributes = {key: value for key,
                              value in kwargs.items() if hasattr(self, key)}
        return updated_attributes

    def run_gradio_chat(self, message: str,
                        history: list,
                        temperature: float,
                        max_new_tokens: int,
                        max_token: int,
                        call_agent: bool,
                        conversation: gr.State,
                        max_round: int = 20,
                        seed: int = None,
                        call_agent_level: int = 0,
                        sub_agent_task: str = None) -> str:
        """
        Generate a streaming response using the llama3-8b model.
        Args:
            message (str): The input message.
            history (list): The conversation history used by ChatInterface.
            temperature (float): The temperature for generating the response.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            str: The generated response.
        """
        print("\033[1;32;40mstart\033[0m")
        print("len(message)", len(message))
        if len(message) <= 10:
            yield "Hi, I am TxAgent, an assistant for answering biomedical questions. Please provide a valid message with a string longer than 10 characters."
            return "Please provide a valid message."
        outputs = []
        outputs_str = ''
        last_outputs = []

        picked_tools_prompt, call_agent_level = self.initialize_tools_prompt(
            call_agent,
            call_agent_level,
            message)

        conversation = self.initialize_conversation(
            message,
            conversation=conversation,
            history=history)
        history = []

        next_round = True
        function_call_messages = []
        current_round = 0
        enable_summary = False
        last_status = {}  # for summary
        token_overflow = False
        if self.enable_checker:
            checker = ReasoningTraceChecker(
                message, conversation, init_index=len(conversation))

        try:
            while next_round and current_round < max_round:
                current_round += 1
                if len(last_outputs) > 0:
                    function_call_messages, picked_tools_prompt, special_tool_call, current_gradio_history = yield from self.run_function_call_stream(
                        last_outputs, return_message=True,
                        existing_tools_prompt=picked_tools_prompt,
                        message_for_call_agent=message,
                        call_agent=call_agent,
                        call_agent_level=call_agent_level,
                        temperature=temperature)
                    history.extend(current_gradio_history)
                    if special_tool_call == 'Finish':
                        yield history
                        next_round = False
                        conversation.extend(function_call_messages)
                        return function_call_messages[0]['content']
                    elif special_tool_call == 'RequireClarification' or special_tool_call == 'DirectResponse':
                        history.append(
                            ChatMessage(role="assistant", content=history[-1].content))
                        yield history
                        next_round = False
                        return history[-1].content
                    if (self.enable_summary or token_overflow) and not call_agent:
                        if token_overflow:
                            print("token_overflow, using summary")
                        enable_summary = True
                    last_status = self.function_result_summary(
                        conversation, status=last_status,
                        enable_summary=enable_summary)
                    if function_call_messages is not None:
                        conversation.extend(function_call_messages)
                        formated_md_function_call_messages = tool_result_format(
                            function_call_messages)
                        yield history
                    else:
                        next_round = False
                        conversation.extend(
                            [{"role": "assistant", "content": ''.join(last_outputs)}])
                        return ''.join(last_outputs).replace("</s>", "")
                if self.enable_checker:
                    good_status, wrong_info = checker.check_conversation()
                    if not good_status:
                        next_round = False
                        print("Internal error in reasoning: " + wrong_info)
                        break
                last_outputs = []
                last_outputs_str, token_overflow = self.llm_infer(
                    messages=conversation,
                    temperature=temperature,
                    tools=picked_tools_prompt,
                    skip_special_tokens=False,
                    max_new_tokens=max_new_tokens,
                    max_token=max_token,
                    seed=seed,
                    check_token_status=True)
                last_thought = last_outputs_str.split("[TOOL_CALLS]")[0]
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'
                if '[FinalAnswer]' in last_thought:
                    final_thought, final_answer = last_thought.split(
                        '[FinalAnswer]')
                    history.append(
                        ChatMessage(role="assistant",
                                    content=final_thought.strip())
                    )
                    yield history
                    history.append(
                        ChatMessage(
                            role="assistant", content="**Answer**:\n"+final_answer.strip())
                    )
                    yield history
                else:
                    history.append(ChatMessage(
                        role="assistant", content=last_thought))
                    yield history

                last_outputs.append(last_outputs_str)

            if self.force_finish:
                last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
                    conversation, temperature, max_new_tokens, max_token, return_full_thought=True)
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'

                final_thought, final_answer = last_outputs_str.split('[FinalAnswer]')
                history.append(
                    ChatMessage(role="assistant",
                                content=final_thought.strip())
                )
                yield history
                history.append(
                    ChatMessage(
                        role="assistant", content="**Answer**:\n"+final_answer.strip())
                )
                yield history
            else:
                yield "The number of rounds exceeds the maximum limit!"

        except Exception as e:
            print(f"Error: {e}")
            if self.force_finish:
                last_outputs_str = self.get_answer_based_on_unfinished_reasoning(
                    conversation,
                    temperature,
                    max_new_tokens,
                    max_token,
                    return_full_thought=True)
                for each in history:
                    if each.metadata is not None:
                        each.metadata['status'] = 'done'

                final_thought, final_answer = last_outputs_str.split(
                    '[FinalAnswer]')
                history.append(
                    ChatMessage(role="assistant",
                                content=final_thought.strip())
                )
                yield history
                history.append(
                    ChatMessage(
                        role="assistant", content="**Answer**:\n"+final_answer.strip())
                )
                yield history
            else:
                return None
