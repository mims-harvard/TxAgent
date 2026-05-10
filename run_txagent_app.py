import random
import datetime
import sys
from txagent import TxAgent
import gradio as gr
import os

os.environ["VLLM_USE_V1"] = "0" # Disable v1 API for now since it does not support logits processors.

# Determine the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["MKL_THREADING_LAYER"] = "GNU"

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)


DESCRIPTION = '''
<div>
<h1 style="text-align: center;">TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools </h1>
</div>
'''
INTRO = """
Precision therapeutics require multimodal adaptive models that provide personalized treatment recommendations. We introduce TxAgent, an AI agent that leverages multi-step reasoning and real-time biomedical knowledge retrieval across a toolbox of 211 expert-curated tools to navigate complex drug interactions, contraindications, and patient-specific treatment strategies, delivering evidence-grounded therapeutic decisions. TxAgent executes goal-oriented tool selection and iterative function calls to solve therapeutic tasks that require deep clinical understanding and cross-source validation. The ToolUniverse consolidates 211 tools linked to trusted sources, including all US FDA-approved drugs since 1939 and validated clinical insights from Open Targets.
"""

LICENSE = """
We welcome your feedback and suggestions to enhance your experience with TxAgent, and if you're interested in collaboration, please email Marinka Zitnik and Shanghua Gao.

### Medical Advice Disclaimer
DISCLAIMER: THIS WEBSITE DOES NOT PROVIDE MEDICAL ADVICE
The information, including but not limited to, text, graphics, images and other material contained on this website are for informational purposes only. No material on this site is intended to be a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health care provider with any questions you may have regarding a medical condition or treatment and before undertaking a new health care regimen, and never disregard professional medical advice or delay in seeking it because of something you have read on this website.
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">TxAgent</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Tips before using TxAgent:</p>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.55;">Please click clear🗑️
 (top-right) to remove previous context before sumbmitting a new question.</p>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.55;">Click retry🔄 (below message)  to get multiple versions of the answer.</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
.small-button button {
    font-size: 12px !important;
    padding: 4px 8px !important;
    height: 6px !important;
    width: 4px !important;
}
.gradio-accordion {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}
"""

chat_css = """
.gr-button { font-size: 20px !important; }  /* Enlarges button icons */
.gr-button svg { width: 32px !important; height: 32px !important; } /* Enlarges SVG icons */
"""

# model_name = '/n/holylfs06/LABS/mzitnik_lab/Lab/shgao/bioagent/bio/alignment-handbook/data_new/L8-qlora-biov49v9v7v16_32k_chat01_merged'
model_name = 'mims-harvard/TxAgent-T1-Llama-3.1-8B'
rag_model_name = 'mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B'

os.environ["TOKENIZERS_PARALLELISM"] = "false"


question_examples = [
    ['Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of moderate hepatic impairment?'],
    ['Given a 50-year-old patient experiencing severe acute pain and considering the use of the newly approved medication, Journavx, how should the dosage be adjusted considering the presence of severe hepatic impairment?'],
    ['A 30-year-old patient is taking Prozac to treat their depression. They were recently diagnosed with WHIM syndrome and require a treatment for that condition as well. Is Xolremdi suitable for this patient, considering contraindications?'],
]

new_tool_files = {
    'new_tool': os.path.join(current_dir, 'data', 'new_tool.json'),
}

agent = TxAgent(model_name,
                rag_model_name,
                tool_files_dict=new_tool_files,
                force_finish=True,
                enable_checker=True,
                step_rag_num=10,
                seed=100,
                additional_default_tools=['DirectResponse', 'RequireClarification'])
agent.init_model()


def update_model_parameters(enable_finish, enable_rag, enable_summary,
                            init_rag_num, step_rag_num, skip_last_k,
                            summary_mode, summary_skip_last_k, summary_context_length, force_finish, seed):
    # Update model instance parameters dynamically
    updated_params = agent.update_parameters(
        enable_finish=enable_finish,
        enable_rag=enable_rag,
        enable_summary=enable_summary,
        init_rag_num=init_rag_num,
        step_rag_num=step_rag_num,
        skip_last_k=skip_last_k,
        summary_mode=summary_mode,
        summary_skip_last_k=summary_skip_last_k,
        summary_context_length=summary_context_length,
        force_finish=force_finish,
        seed=seed,
    )

    return updated_params


def update_seed():
    # Update model instance parameters dynamically
    seed = random.randint(0, 10000)
    updated_params = agent.update_parameters(
        seed=seed,
    )
    return updated_params


def handle_retry(history, retry_data: gr.RetryData, temperature, max_new_tokens, max_tokens, multi_agent, conversation, max_round):
    print("Updated seed:", update_seed())
    new_history = history[:retry_data.index]
    previous_prompt = history[retry_data.index]['content']

    print("previous_prompt", previous_prompt)

    yield from agent.run_gradio_chat(new_history + [{"role": "user", "content": previous_prompt}], temperature, max_new_tokens, max_tokens, multi_agent, conversation, max_round)


PASSWORD = "mypassword"

# Function to check if the password is correct


def check_password(input_password):
    if input_password == PASSWORD:
        return gr.update(visible=True), ""
    else:
        return gr.update(visible=False), "Incorrect password, try again!"


conversation_state = gr.State([])

# Gradio block
chatbot = gr.Chatbot(height=800, placeholder=PLACEHOLDER,
                     label='TxAgent', type="messages",  show_copy_button=True)

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.Markdown(INTRO)
    default_temperature = 0.3
    default_max_new_tokens = 1024
    default_max_tokens = 81920
    default_max_round = 30
    temperature_state = gr.State(value=default_temperature)
    max_new_tokens_state = gr.State(value=default_max_new_tokens)
    max_tokens_state = gr.State(value=default_max_tokens)
    max_round_state = gr.State(value=default_max_round)
    chatbot.retry(handle_retry, chatbot, chatbot, temperature_state, max_new_tokens_state,
                  max_tokens_state, gr.Checkbox(value=False, render=False), conversation_state, max_round_state)

    gr.ChatInterface(
        fn=agent.run_gradio_chat,
        chatbot=chatbot,
        fill_height=True, fill_width=True, stop_btn=True,
        additional_inputs_accordion=gr.Accordion(
            label="⚙️ Inference Parameters", open=False, render=False),
        additional_inputs=[
            temperature_state, max_new_tokens_state, max_tokens_state,
            gr.Checkbox(
                label="Activate multi-agent reasoning mode (it requires additional time but offers a more comprehensive analysis).", value=False, render=False),
            conversation_state,
            max_round_state,
            gr.Number(label="Seed", value=100, render=False)
        ],
        examples=question_examples,
        cache_examples=False,
        css=chat_css,
    )

    with gr.Accordion("Settings", open=False):

        # Define the sliders
        temperature_slider = gr.Slider(
            minimum=0,
            maximum=1,
            step=0.1,
            value=default_temperature,
            label="Temperature"
        )
        max_new_tokens_slider = gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=default_max_new_tokens,
            label="Max new tokens"
        )
        max_tokens_slider = gr.Slider(
            minimum=128,
            maximum=32000,
            step=1,
            value=default_max_tokens,
            label="Max tokens"
        )
        max_round_slider = gr.Slider(
            minimum=0,
            maximum=50,
            step=1,
            value=default_max_round,
            label="Max round")

        # Automatically update states when slider values change
        temperature_slider.change(
            lambda x: x, inputs=temperature_slider, outputs=temperature_state)
        max_new_tokens_slider.change(
            lambda x: x, inputs=max_new_tokens_slider, outputs=max_new_tokens_state)
        max_tokens_slider.change(
            lambda x: x, inputs=max_tokens_slider, outputs=max_tokens_state)
        max_round_slider.change(
            lambda x: x, inputs=max_round_slider, outputs=max_round_state)

        password_input = gr.Textbox(
            label="Enter Password for More Settings", type="password")
        incorrect_message = gr.Textbox(visible=False, interactive=False)
        with gr.Accordion("⚙️ Settings", open=False, visible=False) as protected_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ Model Loading", open=False):
                        model_name_input = gr.Textbox(
                            label="Enter model path", value=model_name)
                        load_model_btn = gr.Button(value="Load Model")
                        load_model_btn.click(
                            agent.load_models, inputs=model_name_input, outputs=gr.Textbox(label="Status"))
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ Functional Parameters", open=False):
                        # Create Gradio components for parameter inputs
                        enable_finish = gr.Checkbox(
                            label="Enable Finish", value=True)
                        enable_rag = gr.Checkbox(
                            label="Enable RAG", value=True)
                        enable_summary = gr.Checkbox(
                            label="Enable Summary", value=False)
                        init_rag_num = gr.Number(
                            label="Initial RAG Num", value=0)
                        step_rag_num = gr.Number(
                            label="Step RAG Num", value=10)
                        skip_last_k = gr.Number(label="Skip Last K", value=0)
                        summary_mode = gr.Textbox(
                            label="Summary Mode", value='step')
                        summary_skip_last_k = gr.Number(
                            label="Summary Skip Last K", value=0)
                        summary_context_length = gr.Number(
                            label="Summary Context Length", value=None)
                        force_finish = gr.Checkbox(
                            label="Force FinalAnswer", value=True)
                        seed = gr.Number(label="Seed", value=100)
                        # Button to submit and update parameters
                        submit_btn = gr.Button("Update Parameters")

                        # Display the updated parameters
                        updated_parameters_output = gr.JSON()

                        # When button is clicked, update parameters
                        submit_btn.click(fn=update_model_parameters,
                                         inputs=[enable_finish, enable_rag, enable_summary, init_rag_num, step_rag_num, skip_last_k,
                                                 summary_mode, summary_skip_last_k, summary_context_length, force_finish, seed],
                                         outputs=updated_parameters_output)
        # Button to submit the password
        submit_button = gr.Button("Submit")

        # When the button is clicked, check if the password is correct
        submit_button.click(
            check_password,
            inputs=password_input,
            outputs=[protected_accordion, incorrect_message]
        )
    gr.Markdown(LICENSE)


if __name__ == "__main__":
    demo.launch(share=True)
