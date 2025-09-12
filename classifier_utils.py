import torch
import torch.nn as nn
from typing import Optional
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from tqdm import tqdm

class classifier(nn.Module):
    def __init__(self,model,num_layers,num_labels):
        super().__init__()
        self.config = model.config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers[:num_layers]
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb
        self.pool = lambda x, mask: (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

        self.linear = nn.Linear(model.config.hidden_size,num_labels).to(model.device)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        cache_position: torch.Tensor = torch.arange(0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        device = hidden_states.device
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = tuple(v.to(device) for v in position_embeddings)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

        hidden_states = self.norm(hidden_states).to(attention_mask.device)
        hidden_states = self.pool(hidden_states,attention_mask)
        out = self.linear(hidden_states)
        
        return out
    

class classifier_training():
    def __init__(self,model) -> None:
        self.model = model
    def training(self,train_loader,test_loader):
        # 1. Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze only the linear head
        for param in self.model.linear.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(self.model.linear.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        device = next(self.model.parameters()).device


        for step,batch in tqdm(enumerate(train_loader, 1)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()

            optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask=attention_mask)
            # import pdb;pdb.set_trace()
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

            # print(f"Step {step} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
        
        test_acc = []
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()
            logits = self.model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()
            test_acc.append(acc)
        print(f"test accuracy is {torch.mean(torch.tensor(test_acc),dtype = torch.float32)}")