import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import tempfile
import json
import os
import shutil
from safetensors.torch import load_model, save_model


class MLPGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(self, mlora_layers, base_size, embd_size):
        super().__init__()
        self.mlora_layers = mlora_layers
        self.base_size = base_size
        self.embd_size = embd_size
        
    def init_generator(self, model):
        # create a NN for each layer
        groups = []
        self.generator_modules = {}
        self.generator_modules["baselinear"] = nn.Linear(self.embd_size, self.base_size)
        
        for layer_name, param in model.named_parameters():
            if layer_name in self.mlora_layers: # Layer chosen for MetaLoRA
                group = layer_name.split(".")[2]
                layer_name = layer_name.replace(".", "_")
                
                if group not in groups:
                    self.generator_modules[group+"_basenn"] = nn.Sequential(
                        nn.Linear(self.base_size, self.base_size*2),
                        nn.ReLU(),
                        nn.Linear(self.base_size*2, self.base_size*2),
                        nn.ReLU(),
                        nn.Linear(self.base_size*2, self.base_size)
                    )
                    self.group_module = {}
                    groups.append(group)
                
                A_size, B_size = param.shape
                self.group_module[layer_name+"_A"] = nn.Linear(self.base_size, A_size)
                self.group_module[layer_name+"_B"] = nn.Linear(self.base_size, B_size)
                self.generator_modules[group+"_layers"] = nn.ModuleDict(self.group_module)
        
        self.generator = nn.ModuleDict(self.generator_modules)

class EmbdModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

    def encode(self, text):
        return self.model.encode([text], show_progress_bar=False)
        
        
class MLoRAmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_name = config["model_name"]
        self.mlora_layers = config["mlora_layers"]
        self.base_size = config["base_size"]
        self.embd_size = config["embd_size"]

        
    def push_to_hub(self, repo, token=None, push_generator=True, push_embd_model=True, private=False):
        # Utiliser un dossier temporaire pour sauvegarder les modèles
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo, exist_ok=True, private=private).repo_id
        
        with tempfile.TemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo
            
            # Pousser le générateur dans un sous-dossier
            if push_generator:
                generator_path = saved_path / "generator"
                generator_path.mkdir(parents=True, exist_ok=True)
                self.generator.save_pretrained(generator_path)
                
                api.upload_folder(
                    folder_path=generator_path,
                    repo_id=repo_id,
                    path_in_repo="generator",
                )
                print(f"Generator pushed to {repo}/generator")
            
            # Pousser l'encoder dans un autre sous-dossier
            
            if push_embd_model:
                embd_model_path = saved_path / "embd_model"
                embd_model_path.mkdir(parents=True, exist_ok=True)
                self.embd_model.save_pretrained(embd_model_path)
                
                api.upload_folder(
                    folder_path=embd_model_path,
                    repo_id=repo_id,
                    path_in_repo="embd_model",
                )
                print(f"EmbdModel pushed to {repo}/embd_model")
            
            ### push config
            config_path = saved_path / "config.json"
            with open(config_path, "w") as config_file:
                json.dump(self.config, config_file, indent=4)

            api.upload_file(
                path_or_fileobj=config_path,
                repo_id=repo_id,
                path_in_repo="config.json",  # Push to the main folder
            )
            print(f"Config pushed to {repo}/config.json")

    @classmethod
    def from_pretrained(cls, repo, token=None, load_generator=True, load_embd_model=True):
        
        with tempfile.TemporaryDirectory() as tmp:
            # load the repo
            snapshot_download(repo_id=repo, local_dir=f"{tmp}/repo/", token=token)
            # create the instance of MLoRAmodel
            config = open(f"{tmp}/repo/config.json")
            config = json.load(config)
            mloramodel = cls(config)
            mloramodel.load_model()
            #shutil.copyfile(f"{tmp}/repo/config.json", f"{tmp}/repo/generator/config.json")
            
            if load_generator:
                mloramodel.generator = MLPGenerator(
                    mlora_layers=config["mlora_layers"], 
                    base_size=config["base_size"], 
                    embd_size=config["embd_size"]
                )
                mloramodel.generator.init_generator(mloramodel.model)
                load_model(mloramodel.generator, f"{tmp}/repo/generator/model.safetensors")
            if load_embd_model:
                mloramodel.embd_model = EmbdModel()
                
        return mloramodel

    def load_model(self, token=None):
        print("load model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=token)
        # set requires grad to False
        for layer_name, param in self.model.named_parameters():
            param.requires_grad = False # define the require_grad to False
        print("model loaded")
    
    
    def init(self, generator=True, embd_model=True):
        if generator:
            self.generator = MLPGenerator(self.mlora_layers, self.base_size, self.embd_size)
            self.generator.init_generator(self.model)
        if embd_model:
            self.embd_model = EmbdModel()
    
    
    def MLoRA(self, base):
        # generate the matrix from base and apply it to the weights
        tp = []
        groups = []
        self.mlora_model = copy.deepcopy(self.model).to(base.device)
        
        for layer_name, param in self.mlora_model.named_parameters():
            
            if layer_name in self.mlora_layers:
                group = layer_name.split(".")[2]
                layer_name_underscore = layer_name.replace(".", "_")
                
                if group not in groups:
                    base_for_group = self.generator.generator[group+"_basenn"](base)
                    groups.append(group)
                    
                A = self.generator.generator[group+"_layers"][layer_name_underscore+"_A"](base_for_group).transpose(0, 1)
                B = self.generator.generator[group+"_layers"][layer_name_underscore+"_B"](base_for_group)
                AB = torch.matmul(A, B)
    
                param.data = param.data + AB
  
    
    def run_MLoRA(self, text, device):
        # run sentences-piece
        embd = torch.tensor(self.embd_model.encode(text)) # (1, self.embd_size)
        base = self.generator.generator["baselinear"](embd.to(device)) # (1, self.base_size)
        # run MLoRA
        self.MLoRA(base)

    
    def forward(self, x):
        # forward model
        out = self.mlora_model.forward(x).logits
        return out
    
    def del_model(self):
        try:
            del self.model
            del self.mlora_model
        except:
            print("mlora or model doesn't exist")
    
    def generate(self, input_ids, text, device, stream=False, tokenizer=None, max_length=50, do_sample=False, temperature=1.0, eos_token_id=2):
        """
        Generate text sequences from input IDs using greedy or sampling-based decoding.

        Parameters:
        - input_ids (torch.Tensor): The input token IDs to start the generation.
        - max_length (int): The maximum length of the generated sequence.
        - do_sample (bool): Whether to use sampling instead of greedy decoding.
        - temperature (float): Used to modulate the next token probabilities if sampling.

        Returns:
        - output_sequences (torch.Tensor): The generated sequence of token IDs.
        """
        # Set model in evaluation mode
        self.run_MLoRA(text, device)
        self.model.eval()

        # Initialize the output sequence with the input ids
        output_sequences = input_ids

        # Loop until max_length is reached
        for _ in range(max_length):
            # Forward pass through the model
            outputs = self.forward(output_sequences)

            # Get the logits for the last token
            logits = outputs[:, -1, :]  # (batch_size, vocab_size)

            if do_sample:
                # Apply temperature and sample the next token from distribution
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding (select the token with the highest probability)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if stream:
                print(tokenizer.decode(next_token[0]), end="")
            
            # Append the next token to the output sequence
            output_sequences = torch.cat([output_sequences, next_token], dim=-1)

            # Stop generation if the next token is the end-of-sequence token (optional)
            if next_token.item() == eos_token_id:
                break
        del self.mlora_model
        return output_sequences