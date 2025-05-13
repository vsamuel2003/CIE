import torch  
from torch import nn
from typing import Optional, Any, Dict
from transformers import AutoModelForCausalLM, LlamaForCausalLM
import re


def remove_non_numeric(string_list):
    """
    Removes all non-numeric characters from each string in a list.
    
    Args:
        string_list: A list of strings to process
        
    Returns:
        A list of strings with only numeric characters
    """
    return [re.sub(r'[^0-9]', '', s) for s in string_list]


class CausalLMWithControlToken(nn.Module):
    """
    A wrapper around a causal language model that allows for controlling generation
    using special control tokens that influence model behavior.
    """

    def __init__(
        self,
        model_name,
        response_ids,
    ):
        """
        Initialize the controlled causal language model.
        
        Args:
            model_name: Name or path of the base model to load
            response_ids: Token IDs that represent the response part of the input
        """
        super().__init__()

        if "gemma" in model_name:
            self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            )

        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )


        self.response_ids = response_ids
        self.config = self.base_model.config

        self.vocab_size, embedding_dim = self.config.vocab_size, self.config.hidden_size
        
        # initialize embeddings for two discrete control tokens (low and high)
        self.control_embeds = nn.Embedding(2, embedding_dim, dtype=torch.bfloat16)
        self.control_embeds.weight.data.normal_(mean=0.0, std=0.02)


        self.ctrl_min, self.ctrl_max = 1, 200
    
    def get_input_embeddings(self):
        """
        Returns the input embeddings of the base model.
        
        Returns:
            The input embedding layer of the base model
        """
        return self.base_model.get_input_embeddings()

    @property
    def generation_config(self):
        """
        Gets the generation configuration of the base model.
        
        Returns:
            The generation configuration
        """
        return self.base_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        """
        Sets the generation configuration of the base model.
        
        Args:
            value: The new generation configuration
        """
        self.base_model.generation_config = value
    
    @property
    def device(self):
        """
        Gets the device on which the model parameters are stored.
        
        Returns:
            The device of the model parameters
        """
        return next(self.base_model.parameters()).device

    
    def gen_labels(self, input_ids, response_ids):
        """
        Generates labels for training by masking out non-response tokens.
        
        Args:
            input_ids: The input token IDs
            response_ids: The token IDs representing the response
            
        Returns:
            A tensor of labels with -100 for tokens that should be ignored in loss calculation
        """
        input_ids = input_ids.clone().detach()
        batch_size = input_ids.shape[0]
        response_ids = response_ids[:, 1:]
        response_len = response_ids.shape[1]
        response_ids = response_ids.to(input_ids.device)

        for i in range(batch_size):
            for j in range(input_ids.shape[1] - response_len + 1):
                if torch.equal(input_ids[i, j:j + response_len], response_ids[0]):
                    input_ids[i, :j + response_len] = -100
                    break
        
        return input_ids
    
    
    def prepare_inputs_for_generation(self, input_ids, verbosity, ctrl_mask):
        """
        Prepares input embeddings by replacing control tokens with control embeddings.
        
        Args:
            input_ids: The input token IDs
            verbosity: The verbosity values to control generation
            ctrl_mask: A mask indicating which positions contain control tokens
            
        Returns:
            The prepared input embeddings with control tokens replaced
        """
        ctrl_normed = (
            (verbosity).clamp(self.ctrl_min, self.ctrl_max)
            - self.ctrl_min
        ) / (self.ctrl_max - self.ctrl_min)


        control_values = torch.stack((ctrl_normed, 1 - ctrl_normed), dim=2).to(
            self.control_embeds.weight.dtype
        )

        control_embeds = control_values @ self.control_embeds.weight
        token_embeds = self.base_model.get_input_embeddings()(input_ids).to(torch.bfloat16)

        
        inputs_embeds = torch.where(ctrl_mask, control_embeds, token_embeds)
        inputs_embeds = inputs_embeds.bfloat16()

        return inputs_embeds


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: bool = True,
        verbosity_mask=None,
        evaluate=False,
        **kwargs: dict[str, Any],
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: The input token IDs
            attention_mask: Mask to avoid attending to padding tokens
            output_hidden_states: Whether to return hidden states
            verbosity_mask: Mask indicating positions with verbosity control tokens
            evaluate: If True, only returns the prepared inputs without running the model
            **kwargs: Additional arguments to pass to the base model
            
        Returns:
            Model outputs or prepared inputs if evaluate=True
        """
        
        if verbosity_mask is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        verbosity = verbosity_mask 
        ctrl_mask = verbosity.clamp(0,1).bool()
        labels = self.gen_labels(input_ids, self.response_ids)
        input_ids = torch.where(ctrl_mask, torch.full_like(input_ids, self.vocab_size - 1), input_ids)


        ctrl_mask = ctrl_mask.unsqueeze(-1)
        model_inputs = self.prepare_inputs_for_generation(input_ids, verbosity, ctrl_mask).to(torch.bfloat16)

        if evaluate:
            return model_inputs

        outputs = self.base_model(
            inputs_embeds= model_inputs,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            labels = labels
        )

        return outputs

    def generate(self, input_ids, **kwargs):
        """
        Generates text using the base model.
        
        Args:
            input_ids: The input token IDs
            **kwargs: Additional arguments to pass to the base model's generate method
            
        Returns:
            Generated token IDs
        """
        return self.base_model.generate(input_ids, **kwargs)