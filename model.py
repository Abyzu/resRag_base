from llama_index.llms.huggingface import HuggingFaceLLM


class LLM(HuggingFaceLLM):
    def __init__(
        self,
        context_window: int = 4096,
        max_new_tokens: int = 2048,
        model_name: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        tokenizer_name="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
        generate_kwargs={"temperature": 0.0, "do_sample": False},
    ) -> None:
        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device_map=device_map,
            model_kwargs=model_kwargs,
            generate_kwargs=generate_kwargs,
        )
