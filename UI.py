import os
import logging
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Проверка доступности CUDA
if torch.cuda.is_available():
    logging.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("CUDA is not available. Using CPU.")

# Загрузка модели и токенизатора
model_name = "VPTQ-community/Qwen2.5-14B-Instruct-v16-k65536-65536-woft"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', local_files_only=True)
    logging.info("Model loaded successfully from local cache.")
except Exception as e:
    logging.error(f"Failed to load model from local cache: {e}")
    logging.info("Attempting to download the model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        logging.info("Model downloaded and loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to download and load the model: {e}")
        raise

def generate_text(prompt, max_length=100):
    try:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length, pad_token_id=tokenizer.pad_token_id)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = generated_text.split("assistant\n")[-1].strip()

        return assistant_response
    except Exception as e:
        logging.error(f"Error in text generation: {str(e)}")
        return f"Произошла ошибка при генерации текста: {str(e)}"

# Создание интерфейса Gradio
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Введите ваш запрос здесь..."),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Максимальная длина генерации")
    ],
    outputs=gr.Textbox(lines=10, label="Сгенерированный текст"),
    title="Qwen2.5-14B Text Generation",
    description="Введите запрос, и модель Qwen2.5-14B сгенерирует текст."
)

# Запуск интерфейса
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)