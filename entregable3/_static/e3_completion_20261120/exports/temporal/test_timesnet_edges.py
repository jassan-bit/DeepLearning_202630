"""Export contract diagnostics only: no data selection, model fitting or recipe changes."""
import itertools,json,sys
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from export_temporal import ROOT,build,FinitePeriodExportBlock,radix2_real_fft8,sha,write

torch.set_num_threads(1);torch.set_num_interop_threads(1)
cp=torch.load(ROOT/'inputs_private/TimesNet.pt',map_location='cpu',weights_only=True)
model=build('TimesNet',cp).eval();original=model.backbone.model[0];adapter=FinitePeriodExportBlock(original).eval()
latent=np.zeros((1,8,32),np.float32)
path=ROOT/'inputs_private/TimesNet_block_edges.onnx'
torch.onnx.export(adapter,torch.from_numpy(latent),str(path),input_names=['latent'],output_names=['latent_output'],opset_version=17,dynamo=False)
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
sess=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
# Generic phase profiles probe tied spectra below the embedding, plus bounded seeded cases.
rng=np.random.default_rng(4321);n=np.arange(8,dtype=np.float64);cases=[('zero_latent',latent.copy()),('constant_latent',np.ones_like(latent)),('alternating_latent',np.broadcast_to(((-1.)**n)[None,:,None],latent.shape).astype(np.float32).copy()),('symmetric_latent',np.broadcast_to(np.asarray([1,2,3,4,4,3,2,1])[None,:,None],latent.shape).astype(np.float32).copy())]
for a,b in [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]:
 v=np.cos(2*np.pi*n*a/8)+np.cos(2*np.pi*n*b/8)
 cases.append((f'equal_cosine_frequencies_{a}_{b}',np.broadcast_to(v[None,:,None],latent.shape).astype(np.float32).copy()))
for j in range(4):cases.append((f'seeded_random_latent_{j}',rng.normal(size=latent.shape).astype(np.float32)))
rows=[]
with torch.inference_mode():
 for name,v in cases:
  x=torch.from_numpy(v);amp=torch.fft.rfft(x,dim=1).abs().mean(0).mean(-1)[1:];real,imag=radix2_real_fft8(x);alt=adapter.spectral(x).mean(0)
  ref=original(x).numpy();adapt=adapter(x).numpy();export=sess.run(None,{'latent':v})[0]
  rows.append({'case':name,'rfft_nonDC_amplitudes':amp.tolist(),'DFT_nonDC_amplitudes':alt.tolist(),'original_topk_indices_nonDC_zero_based':amp.topk(3).indices.tolist(),'butterfly_CPU_topk_indices_nonDC_zero_based':adapter.indices(alt).tolist(),'maximum_FFT_coefficient_error_vs_rFFT':float(abs(torch.complex(real,imag)-torch.fft.rfft(x,dim=1)[:,1:]).max()),'maximum_latent_error_adapter_vs_original':float(abs(adapt-ref).max()),'maximum_latent_error_ORT_vs_original':float(abs(export-ref).max()),'maximum_latent_error_ORT_vs_adapter':float(abs(export-adapt).max())})
write(ROOT/'payload/timesnet/fft_edge_diagnostics.json',{'scope':'Internal latent-block diagnostic, not new model development or training; unscaled latent errors are not output sigma7 errors. Full model market and standardized-input contract results are reported separately.','cases':rows,'mathematical_equivalence':'Radix2 real butterfly factorization gives the nonDC rFFT8 coefficients; all four frequencies remain eligible; floor periods8,4,2,2 evaluate the same shared convolution and input-dependent softmax aggregation. Runtime rank comparisons and a table covering all75 weak orderings preserve pinned CPU Torch2.8 TopK4/k3 tie selection.','floating_point_limit':'Float32 FFT, amplitude reductions and convolution may differ in rounding across runtimes. Generic DFT and ONNX TopK tie mismatches found in the first adapter were corrected through real FFT butterflies and explicit pinned tie order; this finite diagnostic and661market windows are not a proof of output equality for every floating-point input or hardware profile. Overflow/NaN inputs are outside the declared finite bounded serving contract.','new_training_runs':0,'model_checkpoint_sha256':sha(ROOT/'inputs_private/TimesNet.pt'),'adapter_onnx_sha256':sha(ROOT/'payload/timesnet/model.onnx')})
assert all(row['original_topk_indices_nonDC_zero_based']==row['butterfly_CPU_topk_indices_nonDC_zero_based'] for row in rows)
assert max(row['maximum_latent_error_ORT_vs_original'] for row in rows)<=1e-6
for row in rows:print(row['case'],row['maximum_latent_error_ORT_vs_original'],row['original_topk_indices_nonDC_zero_based'],row['butterfly_CPU_topk_indices_nonDC_zero_based'])
