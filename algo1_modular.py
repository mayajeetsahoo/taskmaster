import torch
from tqdm import tqdm


class InterruptExecution(Exception):
    pass

class algo1_functor():
    def __init__(self, model):
        self.co_var = torch.zeros(model.config.intermediate_size, model.config.intermediate_size)
    
    def __call__(self,module,input,output):
        gate = module.gate_proj(input[0])
        up = module.up_proj(input[0])
        hidden = torch.nn.functional.silu(gate) * up
        transpose = torch.transpose(hidden,1,2)
        mul = torch.matmul(transpose,hidden)

        co_var = mul.sum(dim=0)/hidden.shape[0]
        co_var = co_var.to(torch.float32).detach().cpu()
        
        self.co_var = self.co_var + co_var
        raise InterruptExecution()
    
class mode_gpt_algo1():
    def __init__(self) -> None:
        pass
    def prune(self, model, train_loader, sparsity_layer):
        list_layers = list(model.model.layers.named_children())
        # plot_dict = {}
        for layer in tqdm(range(model.config.num_hidden_layers)):
            # print(f'compressing mlp layer of {layer} th decoder layer')

            module = list_layers[layer][1].mlp
            name = list_layers[layer][0]

            module._name = name
            functor = algo1_functor(model)
            hook = module.register_forward_hook(functor)

            for batch in train_loader:
                batch.pop("labels")
                with torch.no_grad():
                    try:
                        _ = model(**batch)
                    except InterruptExecution:
                        pass
            #     break
            # with torch.no_grad():
            #     try:
            #         _ = model(**data)
            #     except InterruptExecution:
            #         pass
            hook.remove()
            ## pruning

            torch.cuda.empty_cache()
            co_var = functor.co_var

            I = torch.eye(co_var.size(0), device=co_var.device, dtype=co_var.dtype)
            A = (co_var+I)
            score_mat = torch.matmul(co_var,torch.inverse(A))
            scores_vector = torch.diagonal(score_mat)
            # plot_dict[name]=scores_vector

            topk_scores, topk_indices = torch.topk(scores_vector, int((1-sparsity_layer[int(module._name)])*co_var.size(0)), largest=True)
            topk_indices = topk_indices.sort().values
            d_int = scores_vector.shape[0]
            S_k = torch.zeros((d_int, int((1-sparsity_layer[int(module._name)])*co_var.size(0))), dtype=torch.float32)
            S_k[topk_indices, torch.arange(int((1-sparsity_layer[int(module._name)])*co_var.size(0)))] = 1.0
            W_U_k = torch.concat([module.up_proj.weight,module.gate_proj.weight],dim=1).T.to(torch.float32) @ S_k.to(module.gate_proj.weight.device) 

            Mat = (S_k.T @ co_var @ S_k).to(torch.float64)
            W_D_k = torch.linalg.pinv(Mat).to(torch.float32) @ (S_k.T @ co_var @ module.down_proj.weight.data.T.cpu())
            
            new_up_proj = W_U_k.T[:,:model.config.hidden_size]
            new_gate_proj = W_U_k.T[:,model.config.hidden_size:]
            module.up_proj.weight.data = new_up_proj.to(module.up_proj.weight.device)
            module.gate_proj.weight.data = new_gate_proj.to(module.gate_proj.weight.device)

            module.down_proj.weight.data = W_D_k.T.to(module.down_proj.weight.device)
        del model