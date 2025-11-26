import dspy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LocalPhi(dspy.LM):
    """
    A custom DSPy LM provider that runs Microsoft Phi-3.5 locally 
    using Hugging Face Transformers.
    """
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", max_tokens=1000):
        # 1. Initialize Parent
        super().__init__(model=model_name)
        
        print(f"Initializing local pipeline for {model_name}...")
        
        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 3. Load Model
        # device_map="auto" efficiently handles GPU/CPU offloading
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype="auto", 
            device_map="auto" 
        )

        # 4. Create Pipeline
        # return_full_text=False is critical so DSPy doesn't get the prompt repeated back
        self.pipe = pipeline(
            "text-generation",
            model=self.hf_model, 
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            return_full_text=False 
        )
        
        # Default configs
        self.default_kwargs = {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "do_sample": False
        }

    def basic_request(self, prompt: str, **kwargs):
        """
        Generates text. DSPy calls this method with a string 'prompt' 
        and optional kwargs (like temperature).
        """
        
        # --- A. Prompt Formatting (Critical for Phi-3) ---
        # DSPy usually sends a raw string. Phi-3 requires specific Chat tags.
        # If the prompt doesn't look like a chat template, we wrap it.
        final_prompt = prompt
        if "<|user|>" not in prompt and "<|assistant|>" not in prompt:
            # We treat the DSPy prompt as the user message
            messages = [{"role": "user", "content": prompt}]
            final_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # --- B. Handle Kwargs ---
        # Merge defaults with run-time kwargs
        gen_kwargs = {**self.default_kwargs, **kwargs}
        
        # Remove DSPy-specific args that break HF pipelines
        gen_kwargs.pop('n', None)       # HF uses num_return_sequences
        gen_kwargs.pop('messages', None) # We are passing a string, not a list
        
        # --- C. Generate ---
        try:
            output = self.pipe(final_prompt, **gen_kwargs)
            generated_text = output[0]['generated_text']
            
            # DSPy expects a LIST of strings
            return [generated_text]
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return [""]

    def __call__(self, prompt, **kwargs):
        # Helper to ensure direct calls work if used outside DSPy flow
        return self.basic_request(prompt, **kwargs)