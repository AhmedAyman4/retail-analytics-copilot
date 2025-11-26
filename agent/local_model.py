import dspy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LocalPhi(dspy.LM):
    """
    A custom DSPy LM provider that runs Microsoft Phi-3.5 locally 
    using Hugging Face Transformers.
    """
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", max_tokens=1000):
        # Pass the string name to the parent class.
        super().__init__(model=model_name)
        
        print(f"Initializing local pipeline for {model_name}...")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {device}")

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Store as self.hf_model to avoid conflict with dspy.LM.model (which must be a string)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device
        )

        # Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.hf_model, 
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            return_full_text=False
        )
        
        # Configuration
        self.kwargs = {
            "temperature": 0.0,
            "max_new_tokens": max_tokens,
            "do_sample": False # Deterministic for agents
        }

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        Override __call__ to strictly enforce local execution.
        We handle both 'prompt' (string) and 'messages' (list) inputs 
        to satisfy different DSPy calling patterns.
        """
        # 1. Handle "messages" (Chat format) if "prompt" is missing
        if prompt is None and messages is not None:
            # Use tokenizer's chat template for correct Phi-3 formatting
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback manual construction if template fails
                prompt = ""
                for m in messages:
                    role = m.get('role', 'user') or 'user'
                    content = m.get('content', '')
                    prompt += f"<|{role}|>\n{content}<|end|>\n"
                prompt += "<|assistant|>\n"
            
        # 2. Fallback if both are missing
        if prompt is None:
            prompt = ""

        # 3. Delegate to basic_request
        return self.basic_request(prompt, **kwargs)

    def basic_request(self, prompt, **kwargs):
        """
        The method that actually runs the pipeline.
        """
        # Merge default kwargs with request kwargs
        gen_kwargs = {**self.kwargs, **kwargs}
        
        # Remove DSPy-specific args that transformers doesn't like
        gen_kwargs.pop('n', None) 
        
        try:
            if not prompt:
                return [""]
                
            output = self.pipe(prompt, **gen_kwargs)
            generated_text = output[0]['generated_text']
            
            # Return list of completions (DSPy expects a list)
            return [generated_text]
        except Exception as e:
            print(f"Error during generation: {e}")
            return [""]