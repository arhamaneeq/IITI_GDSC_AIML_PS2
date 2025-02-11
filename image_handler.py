from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    # Convert image bytes to base64
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")

    # Return base64 string with image header
    return "data:image/jpeg;base64," + encoded_string

def handle_image(image_bytes, user_message):
    # Load the model
    chat_handler = Llava15ChatHandler(clip_model_path = "models\llava\mmproj-model-f16.gguf")
    llm = Llama(
        model_path = "models\llava\ggml-model-q4_k.gguf",
        chat_handler = chat_handler,
        logits_all = True, 
        n_ctx = 1024 
    )

    image_base64 = convert_bytes_to_base64(image_bytes)

    # Generate response
    output = llm.create_chat_completion(
        messages = [
            {
                "role" : "system",
                "content" : "You are an assistant named Raggy. You perfectly describe and understand images, however, you are immensely sarcastic, and try to sound like a highbrow art critic."
            }, {
                "role" : "user",
                "content" : [
                    {
                        "type" : "image_url",
                        "image_url" : {"url" : image_base64}
                    }, {
                        "type" : "text",
                        "text" : user_message
                    }
                ]
            }
        ]
    )

    return output