"""
Reprogram LLM with a input transformation and output mapping.

"""
import numpy as np
import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

from huggingface_hub import login
login(token="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")


class Model(nn.Module):
	def __init__(self, configs):
		super(Model, self).__init__()
		
		# load LLM
		model_name_or_path = "meta-llama/Llama-2-7b-hf"
		self.llama_config = LlamaConfig.from_pretrained(model_name_or_path)
		self.llama_config.num_hidden_layers = configs["llm_layers"]
		self.llama_config.output_attentions = True
		self.llama_config.output_hidden_states = True
		self.llama = LlamaModel.from_pretrained(
			model_name_or_path,
			# trust_remote_code=True,
			# local_files_only=True,
			config=self.llama_config,
			load_in_4bit=True
		)
	
		# freeze LLM
		for param in self.llama.parameters():
			param.requires_grad = False
		
		self.embed = ConvEmbedding(configs["channels"])
		self.input_transformation = InputTransformation(configs["IT"]["in_features"], configs["IT"]["out_features"])
		self.output_mapping = OutputMapping(configs["OM"]["in_features"], configs["OM"]["out_features"])
		
	def forward(self, x):
		x = self.embed(x)
		x = self.input_transformation(x)
		x = self.llama(inputs_embeds=x).last_hidden_state
		x = self.output_mapping(x)
		return x
	
	
class InputTransformation(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear = None
	
	def forward(self, x):
		x = self.flatten(x)
		if self.linear is None:
			dff = x.size()[-1]
			self.linear = nn.Linear(dff, 64)
		x = self.linear(x)
		return x[:, :, np.newaxis]
	
	
class OutputMapping(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear = None
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, x):
		x = self.flatten(x)
		if self.linear is None:
			dff = x.size()[-1]
			self.linear = nn.Linear(dff, 5)
		x = self.linear(x)
		x = self.softmax(x)
		return x
	

class ConvEmbedding(nn.Module):
	def __init__(self, channels=[1, 32, 64]):
		super().__init__()
		self.channels = channels
	
	def forward(self, x):
		for ch_in, ch_out in zip(self.channels[: -1], self.channels[1:]):
			x = nn.Conv2d(ch_in, ch_out, (3, 3))(x)
			x = nn.AvgPool2d((2, 2))(x)
		return x

# self.word_embeddings = self.llama.get_input_embeddings().weight
# self.vocab_size = self.word_embeddings.shape[0]
# self.num_tokens = 1000

	