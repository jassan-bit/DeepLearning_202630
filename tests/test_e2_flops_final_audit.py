"""Independent arithmetic checks for the synthetic core-FLOP convention."""
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_flops_final_audit import core_flops, audit_model

class ForwardAdapter(nn.Module):
    def __init__(self,body,transform=lambda x:x):
        super().__init__();self.body=body;self.transform=transform
    def forward(self,x):
        value=self.body(self.transform(x))
        if isinstance(value,tuple):value=value[0]
        return value.reshape(len(x),-1).sum(-1)

class CoreFLOPTests(unittest.TestCase):
    def test_linear_mac_convention_excludes_bias(self):
        x=torch.zeros(1,7,5)
        model=ForwardAdapter(nn.Linear(35,16),lambda a:a.flatten(1)).eval()
        self.assertEqual(audit_model(model,x)["core_flops"],2*35*16)

    def test_grouped_padded_strided_convolution_uses_output_geometry(self):
        model=ForwardAdapter(nn.Conv1d(6,8,3,stride=2,padding=1,groups=2)).eval()
        x=torch.zeros(1,6,7)
        self.assertEqual(audit_model(model,x)["core_flops"],2*8*4*3*3)

    def test_lstm_gates_match_independent_closed_form(self):
        # Seven steps: four gates, input5, hidden32, input and recurrent products.
        model=ForwardAdapter(nn.LSTM(5,32,batch_first=True)).eval()
        self.assertEqual(audit_model(model,torch.zeros(1,7,5))["core_flops"],2*7*4*32*(5+32))

    def test_gru_gates_match_independent_closed_form(self):
        model=ForwardAdapter(nn.GRU(5,32,batch_first=True)).eval()
        self.assertEqual(audit_model(model,torch.zeros(1,7,5))["core_flops"],2*7*3*32*(5+32))

    def test_exclusions_are_exposed_and_transpose_conv_is_rejected(self):
        model=ForwardAdapter(nn.Sequential(nn.Linear(5,2),nn.Sigmoid())).eval()
        result=audit_model(model,torch.zeros(1,5))
        self.assertEqual(result["core_flops"],20)
        self.assertFalse(result["operators"]["aten.sigmoid.default"]["counted_as_core"])
        args=(torch.zeros(1,2,7),torch.zeros(2,3,3),None,[1],[0],[1],True)
        with self.assertRaises(ValueError):core_flops("aten.convolution.default",args,torch.zeros(1,3,9))

if __name__=="__main__":unittest.main()
