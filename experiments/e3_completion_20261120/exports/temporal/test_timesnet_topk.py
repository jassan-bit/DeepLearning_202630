"""Exhaustive finite order-type contract for the CPU TopK4/k3 export adapter."""
import itertools
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from export_temporal import ROOT,frozen_topk4_rank_table,write,sha

class RankTopK(nn.Module):
 def __init__(self):
  super().__init__();self.register_buffer('table',frozen_topk4_rank_table());self.register_buffer('key_weights',torch.tensor([1,4,16,64],dtype=torch.long))
 def forward(self,x):
  rank=(x[:,None]>x[None,:]).sum(1);key=(rank*self.key_weights).sum();return self.table[key]

torch.set_num_threads(1);model=RankTopK().eval();path=ROOT/'inputs_private/TimesNet_topk_contract.onnx'
torch.onnx.export(model,torch.ones(4),str(path),input_names=['amplitude'],output_names=['indices'],opset_version=17,dynamo=False)
opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
failures=[];keys=set()
for values in itertools.product(range(4),repeat=4):
 x=torch.tensor(values,dtype=torch.float32);reference=x.topk(3).indices.numpy();actual=session.run(None,{'amplitude':x.numpy()})[0]
 key=int(((x[:,None]>x[None,:]).sum(1)*model.key_weights).sum());keys.add(key)
 if not np.array_equal(reference,actual):failures.append({'input':values,'reference':reference.tolist(),'export':actual.tolist()})
report={'status':'passed' if not failures else 'failed','scope':'Exact finite order/equality pattern contract for pinned Torch2.8 CPU TopK of four finite amplitudes selecting largest3 sorted. Not a claim of runtime-independent TopK tie order.','numeric_assignments_checked':256,'unique_weak_orderings_checked':len(keys),'failing_cases':failures,'rank_table_sha256':__import__('hashlib').sha256(model.table.numpy().tobytes()).hexdigest(),'test_graph_sha256':sha(path),'no_epsilon_or_data_selected_thresholds':True,'new_training_runs':0}
write(ROOT/'payload/timesnet/topk_order_contract.json',report);print(report)

assert not failures and len(keys)==75
