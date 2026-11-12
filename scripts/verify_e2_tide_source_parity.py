"""Execute the pinned official TensorFlow TiDE graph and compare its gradients.

Requires separate existing PyTorch and TensorFlow(+tf-keras,tqdm) interpreters.
This verifier installs nothing and uses only deterministic synthetic inputs,
shared test weights and one forward/backward pass. No optimizer is constructed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def torch_fixture(directory, source_root):
    import numpy as np
    import torch
    from e2_published_tide import build_tide
    model = build_tide(seed=271201, source_roots={"TiDE":source_root}).backbone.eval()
    rng = np.random.RandomState(5)
    x = torch.tensor(rng.randn(10,7).astype('float32'), requires_grad=True)
    ids = torch.arange(5).repeat(2)
    target = torch.tensor(rng.randn(10,1).astype('float32'))
    output = model(x, ids)
    loss = (output-target).square().mean()
    loss.backward()
    np.savez(directory/'input_weights.npz',x=x.detach().numpy(),ids=ids.numpy(),target=target.numpy(),
             **{k:v.detach().numpy() for k,v in model.state_dict().items()})
    np.savez(directory/'torch_result.npz',output=output.detach().numpy(),input_gradient=x.grad.numpy(),
             loss=np.array(loss.item()),**{'grad_'+k:p.grad.detach().numpy() for k,p in model.named_parameters()})
    (directory/'torch_runtime.json').write_text(json.dumps({'torch':str(torch.__version__),
        'python':sys.version.split()[0], 'parameters':sum(p.numel() for p in model.parameters())}))


def official_tensorflow_fixture(directory, source_root):
    import numpy as np
    import tensorflow as tf
    import tf_keras
    spec = importlib.util.spec_from_file_location('official_tide',source_root/'tide/models.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    fixture = np.load(directory/'input_weights.npz')
    config = dict(hidden_dims=[256,256],time_encoder_dims=[64,4],decoder_output_dim=4,final_decoder_hidden=64)
    model = author.TideModel(config,1,[],5,transform=False,layer_norm=False,dropout_rate=0.)
    x = tf.Variable(fixture['x'])
    ids, target = tf.constant(fixture['ids']), tf.constant(fixture['target'])
    inputs = ((x,tf.zeros((0,7)),tf.zeros((0,7),dtype=tf.int32)),
              (tf.zeros((0,1)),tf.zeros((0,1),dtype=tf.int32)),ids)
    model(inputs,training=False)
    variables={}
    for prefix, block in [('encoder',model.encoder.layers[0]),('decoder',model.decoder.layers[0]),
                          ('time_encoder',model.time_encoder.layers[0]),('final_decoder',model.final_decoder)]:
        for sub in ('lin_a','lin_b','lin_res'):
            layer=getattr(block,sub)
            name=prefix+'.'+sub
            layer.set_weights([fixture[name+'.weight'].T, fixture[name+'.bias']])
            variables[name+'.weight']=(layer.kernel,True)
            variables[name+'.bias']=(layer.bias,False)
    model.linear.set_weights([fixture['linear.weight'].T,fixture['linear.bias']])
    variables['linear.weight']=(model.linear.kernel,True)
    variables['linear.bias']=(model.linear.bias,False)
    model.ts_embs.set_weights([fixture['ts_embs.weight']])
    variables['ts_embs.weight']=(model.ts_embs.embeddings,False)
    with tf.GradientTape() as tape:
        output=model(inputs,training=False)
        loss=tf.reduce_mean(tf.square(output-target))
    gradients=tape.gradient(loss,[x]+[v[0] for v in variables.values()])
    if any(g is None for g in gradients):
        raise RuntimeError('Official source returned missing gradients; use TF_USE_LEGACY_KERAS=1 and compatible tf-keras')
    result={'output':output.numpy(),'loss':np.array(loss.numpy()),'input_gradient':gradients[0].numpy()}
    for (name,(_,transpose)),gradient in zip(variables.items(),gradients[1:]):
        value=tf.convert_to_tensor(gradient).numpy()
        result['grad_'+name]=value.T if transpose else value
    np.savez(directory/'tf_result.npz',**result)
    expected=np.load(directory/'torch_result.npz')
    comparisons=[]
    for name,value in result.items():
        other=expected[name]
        comparisons.append({'tensor':name,'shape':list(value.shape),
            'max_absolute_error':float(np.max(np.abs(value-other))) if value.size else 0.,
            'allclose':bool(np.allclose(value,other,atol=2e-6,rtol=2e-5))})
    receipt={'version':'e2-tide-official-source-parity-v1', 'tensorflow':tf.__version__,
        'tf_keras':tf_keras.__version__, 'torch_runtime':json.loads((directory/'torch_runtime.json').read_text()),
        'official_source_executed':True,'scope':'Source-default TiDE body L7/H1/r0/num_ts5, FP32 CPU shared weights; scalar task readout tested separately',
        'forward_loss_input_and_parameter_gradients':True,'atol':2e-6,'rtol':2e-5,
        'test_batch_scalar_series':10,'comparisons':comparisons,'all_passed':all(row['allclose'] for row in comparisons),
        'parameter_gradient_tensors':len(variables),'optimizer_constructed':False,'optimizer_steps':0,
        'market_values_read':False,'source_initialization_draws_bitwise_equivalent_claimed':False,
        'initialization_policy':'Both source Dense and port use Glorot-uniform kernels/zero biases, embedding uniform(-.05,.05); framework RNG streams are not claimed identical',
        'GPU_kernel_parity_claimed':False}
    (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--torch-python',default=sys.executable)
    p.add_argument('--tensorflow-python')
    p.add_argument('--output',type=Path)
    p.add_argument('--child',choices=('torch','tensorflow'))
    p.add_argument('--directory',type=Path)
    args=p.parse_args(argv)
    if args.child:
        if args.directory is None:raise ValueError('Child work directory required')
        (torch_fixture if args.child=='torch' else official_tensorflow_fixture)(args.directory,args.source_root)
        return 0
    if not args.tensorflow_python or args.output is None:
        p.error('Supply explicit existing --tensorflow-python and --output')
    from e2_published_tide import verify_tide_sources,RECIPE_ID
    source=verify_tide_sources({'TiDE':args.source_root})
    with tempfile.TemporaryDirectory(prefix='e2-tide-parity-') as temporary:
        directory=Path(temporary)
        env=dict(os.environ,TF_USE_LEGACY_KERAS='1',TF_ENABLE_ONEDNN_OPTS='0',
                 TF_CPP_MIN_LOG_LEVEL='2',MPLCONFIGDIR=str(directory/'matplotlib'), CUDA_VISIBLE_DEVICES='')
        for key in ('GITHUB_TOKEN','GH_TOKEN'):env.pop(key,None)
        for mode,python in [('torch',args.torch_python),('tensorflow',args.tensorflow_python)]:
            subprocess.run([python,str(Path(__file__).resolve()),'--source-root',str(args.source_root),
                '--child',mode,'--directory',str(directory)],env=env,check=True)
        receipt=json.loads((directory/'receipt.json').read_text())
    receipt.update(recipe_id=RECIPE_ID,source=source,
                   verification_implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   port_implementation_sha256=sha256(Path(__file__).with_name('e2_published_tide.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'all_passed':receipt['all_passed'],'parameter_gradient_tensors':receipt['parameter_gradient_tensors'],
                      'optimizer_steps':0,'output':str(args.output)}))
    return 0 if receipt['all_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
