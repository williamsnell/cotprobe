import re
import torch as t
import polars as rs
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

device = "cuda" if t.cuda.is_available() else "cpu"

class ProbeQwen(t.nn.Module):
    def __init__(self, base_model_name, out_features=10, load_in_8bit=False):
        super().__init__()
        # In qwen, .model is everything up until the unembed.
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, load_in_8bit=load_in_8bit)
        self.base_model = base_model.model
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Simple linear probe for now
        in_features = base_model.lm_head.in_features
        self.probe = t.nn.Linear(in_features, out_features)

        self.to(device)

    def forward(self, x):
        residual_stream = self.base_model(x).last_hidden_state
        return self.probe(residual_stream)
        

    def from_text(self, text: str):
        tokens = self.tokenizer(text, return_tensors='pt', padding=True).to(device)
        return self.forward(tokens['input_ids'])        

if __name__ == '__main__':
    assert device == "cuda"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    probed = ProbeQwen(model_name, load_in_8bit=False)

    # Load in some data

    df = rs.read_ipc(Path("responses.arrow"))
    
    prompt = df['text'][0]

    print(probed.from_text(df['text'][0]))
