import dspy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LocalPhi(dspy.LM):
    """
    A custom DSPy LM provider that runs Microsoft Phi-3.5 locally 
    using Hugging Face Transformers.
    """
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", max_tokens=1000):
        # 1. Pass the string name to the parent class. 
        # DSPy stores this in self.model and expects it to remain a string.
        super().__init__(model=model_name)
        
        print(f"Initializing local pipeline for {model_name}...")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on device: {device}")

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 2. FIX: Assign the loaded model object to 'self.hf_model' instead of 'self.model'
        # This prevents the "AttributeError: object has no attribute 'split'"
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device
        )

        # Create Pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.hf_model, # Use the renamed attribute
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

    def basic_request(self, prompt: str, **kwargs):
        """
        The method DSPy calls to generate text.
        """
        # Merge default kwargs with request kwargs
        gen_kwargs = {**self.kwargs, **kwargs}
        
        # Remove DSPy-specific args that transformers doesn't like
        gen_kwargs.pop('n', None) 
        
        try:
            # Phi-3.5 works best with a chat template, but pure string prompting is supported.
            output = self.pipe(prompt, **gen_kwargs)
            generated_text = output[0]['generated_text']
            
            # Return list of completions (DSPy expects a list)
            return [generated_text]
        except Exception as e:
            print(f"Error during generation: {e}")
            return [""]

    def __call__(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)