"""Auditable synthetic forward MAC/FLOP counts for the 13 frozen E2 bodies.

This is an operation-count audit, never market inference, fitting, or a CPU
latency benchmark. A MAC counts as two FLOPs; only dense matrix products and
convolutions are in that common core convention. Every other observed ATen
operator is exported so that the core sum cannot masquerade as all-kernel FLOPs.
The scientific constructor and immutable author-source validators are reused.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import subprocess
import sys

VERSION = "e2-frozen-core-flops-audit-v1"
SCIENTIFIC_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
MODEL_ORDER = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
               "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
               "MLP16", "LSTM32", "GRU32")
MATRIX_OPS = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default", "aten.mv.default", "aten.dot.default"}
CONV_OPS = {"aten.convolution.default", "aten._convolution.default"}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def core_flops(op, args, output):
    """Dense 2*MAC convention, independent of backend latency/kernel fusion.

    Bias, padding comparisons, nonlinearity, reductions and FFT are outside
    this deliberately named core convention. Convolution counts full padded
    kernels, the standard dense algorithmic convention, not sparse zero skips.
    """
    if op == "aten.mm.default":
        a,b=args[:2]; return 2*int(a.shape[0])*int(a.shape[1])*int(b.shape[1])
    if op == "aten.addmm.default":
        a,b=args[1:3]; return 2*int(a.shape[0])*int(a.shape[1])*int(b.shape[1])
    if op == "aten.bmm.default":
        a,b=args[:2]; return 2*int(a.shape[0])*int(a.shape[1])*int(a.shape[2])*int(b.shape[2])
    if op == "aten.mv.default":
        a=args[0]; return 2*int(a.shape[0])*int(a.shape[1])
    if op == "aten.dot.default": return 2*int(args[0].numel())
    if op in CONV_OPS:
        if bool(args[6]):
            raise ValueError("Transposed convolution needs a separately validated formula")
        weight=args[1]
        return 2*int(output.numel())*prod(int(x) for x in weight.shape[1:])
    return None


def audit_model(model, x):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    class Census(TorchDispatchMode):
        def __init__(self): self.rows={}
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out=func(*args,**(kwargs or {})); op=str(func)
            count=core_flops(op,args,out)
            row=self.rows.setdefault(op,{"calls":0,"core_flops":0,"counted_as_core":count is not None,
                                          "input_shapes_examples":[]})
            row["calls"]+=1; row["core_flops"]+=count or 0
            shapes=[list(a.shape) for a in args if isinstance(a,torch.Tensor)]
            if shapes not in row["input_shapes_examples"] and len(row["input_shapes_examples"])<4:
                row["input_shapes_examples"].append(shapes)
            return out
    census=Census()
    # CPU MKLDNN recurrent fusion is disabled to expose the exact recurrent
    # gate matrix products to dispatch. No scientific model or tensor changes.
    with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False), census:
        value=model(x)
    if tuple(value.shape)!=(1,) or not bool(torch.isfinite(value).all()):
        raise ValueError("The admitted D7 scalar predictor did not produce a finite scalar")
    # Never silently miss an opaque multiplication-heavy fused kernel.
    suspects=[op for op in census.rows if any(tag in op for tag in
        ("lstm", "gru", "rnn", "scaled_dot_product", "linear", "matmul", "einsum", "convolution", "addmm", "bmm", "mm."))
        and op not in MATRIX_OPS|CONV_OPS]
    if suspects: raise ValueError("Uncounted fused matrix/kernel operators: "+str(suspects))
    return {"core_flops":sum(r["core_flops"] for r in census.rows.values()),
            "operators":census.rows,"output_shape":list(value.shape)}


def run(scientific_source, architecture_inputs, source_roots_path, destination):
    import torch
    torch.set_num_threads(1)
    source=Path(scientific_source).resolve(); destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source/"scripts"))
    import e2_reduced_fit as factory
    inputs=json.loads(Path(architecture_inputs).read_text())
    rows={r["model"]:r for r in inputs}
    if set(rows)!=set(MODEL_ORDER): raise ValueError("Exactly thirteen frozen neural recipes are required")
    roots=json.loads(Path(source_roots_path).read_text())
    probes={"zeros":torch.zeros(1,7,5), "linear":torch.linspace(-.5,.5,35).view(1,7,5),
            "sinusoid":torch.sin(torch.arange(35,dtype=torch.float32)).view(1,7,5)}
    result=[]
    for name in MODEL_ORDER:
        row=rows[name]; trial=row["trial"]
        if Path(row["result_path"]).exists() and digest(row["result_path"])!=row["result_sha256"]:
            raise ValueError("Frozen source receipt hash differs for "+name)
        model=factory._builder(name,trial,271801,roots).eval()
        parameters=sum(p.numel() for p in model.parameters())
        if parameters!=row["architecture"]["actual_parameter_count"]:
            raise ValueError("Frozen parameter count differs for "+name)
        profiles={key:audit_model(model,value) for key,value in probes.items()}
        costs=[p["core_flops"] for p in profiles.values()]
        receipt={"model":name,"recipe_id":trial["recipe_id"],"winner_trial_id":trial["trial_id"],
            "registered_parameters":parameters,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "core_forward_flops_min":min(costs),"core_forward_flops_max":max(costs),
            "core_MACs_min":min(costs)//2,"core_MACs_max":max(costs)//2,
            "synthetic_probes":profiles,"full_all_kernel_flops_measured":False,
            "frozen_recipe_receipt_path":row["result_path"],"frozen_recipe_receipt_sha256":row["result_sha256"],
            "architecture_metadata":model.architecture_metadata}
        result.append(receipt)
        print(json.dumps({"model":name,"parameters":parameters,"core_flops_range":[min(costs),max(costs)]}),flush=True)
    scripts=("e2_reduced_fit.py","e2_published_registry.py","e2_training_models.py","e2_recipe_models.py",
             "e2_published_periodic.py","e2_published_patchtst.py","e2_published_sae.py","e2_published_tide.py",
             "e2_final_models_mlp.py","e2_final_models_attention.py","dl_models.py","dl_models_gru_lstm_tcn.py")
    source_receipts={}
    for relative in scripts:
        path=source/"scripts"/relative
        original=subprocess.run(["git","-C",str(source),"show",SCIENTIFIC_COMMIT+":scripts/"+relative],capture_output=True,check=True).stdout
        if sha256(original).hexdigest()!=digest(path):
            raise ValueError("Scientific constructor differs from pinned original: "+relative)
        source_receipts["scripts/"+relative]=digest(path)
    document={"version":VERSION,"scientific_commit":SCIENTIFIC_COMMIT,"constructor_sha256":source_receipts,
        "architecture_inputs_sha256":digest(architecture_inputs),
        "runtime":{"torch":torch.__version__,"device":"cpu","dtype":"float32","mkldnn_fused_recurrence":False},
        "scope":{"batch_size":1,"input_shape":[1,7,5],"mode":"eval","synthetic_inputs_only":True,
                 "trained_checkpoint_loaded":False,"market_data_read":False,"optimizer_created":False,
                 "training_performed":False,"new_budget_credit":False,"latency_or_energy_comparison":False},
        "convention":{"one_MAC_in_FLOPs":2,"included":"all dispatched dense mm/addmm/bmm/mv/dot and regular convolution products, including recurrent gate matrices and attention QK/AV",
                      "excluded":"bias additions, all scalar elementwise operations, activations, normalization, reductions/pooling, comparisons/sorting/topk, FFT, indexing and memory movement",
                      "padding":"full dense padded kernels; no zero-skipping or backend-specific pruning",
                      "recurrent":"LSTM/GRU decomposed by PyTorch with MKLDNN disabled; all gate matrix products counted, elementwise gate operations excluded",
                      "TimesNet":"full unpruned published convolution kernels; FFT cost excluded. B1 and length8 imply non-DC periods8,4,2,2; all divide8, so each three-branch block has invariant convolution area under the dense convention",
                      "synthetic_probes_do_not_exhaust_value_dependent_routes":True,
                      "weights":"fresh seeded factory weights, never trained checkpoint weights",
                      "all_kernel_total_claimed":False,"GPU_speedup_claimed":False},
        "classical_scope":{"Persistence":"neural FLOPs not applicable; history lookup", "SeasonalNaive":"neural FLOPs not applicable; history lookup",
                           "ARIMA":"not measured by neural ATen convention; state update/forecast operation count depends on fitted orders and causal updating"},
        "models":result}
    (destination/"flops_audit.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n")
    import csv
    with (destination/"flops_by_model.csv").open("w",newline="") as f:
        fields=["model","registered_parameters","trainable_parameters","core_MACs_min","core_MACs_max","core_forward_flops_min","core_forward_flops_max","winner_trial_id","full_all_kernel_flops_measured"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in result)
    return document


def write_pareto(document, benchmark, destination):
    """Descriptive frozen-test Pareto; no model selection or inference."""
    import csv
    destination=Path(destination)
    metrics={r["model"]:r for r in csv.DictReader(Path(benchmark).open())}
    rows=[{"model":m["model"],"RMSE_frozen_A2":float(metrics[m["model"]]["RMSE"]),
           "core_forward_flops":m["core_forward_flops_max"],"registered_parameters":m["registered_parameters"]}
          for m in document["models"]]
    for x in rows:
        x["pareto_core_flops_RMSE"]=not any(
            y["core_forward_flops"]<=x["core_forward_flops"] and y["RMSE_frozen_A2"]<=x["RMSE_frozen_A2"]
            and (y["core_forward_flops"]<x["core_forward_flops"] or y["RMSE_frozen_A2"]<x["RMSE_frozen_A2"])
            for y in rows)
    with (destination/"pareto_core_flops_rmse.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt={"benchmark_A2_sha256":digest(benchmark),"posthoc_descriptive_only":True,
             "model_reselection_performed":False,"domination_is_not_statistical_significance":True,
             "scope":"thirteen neural systems, core FLOPs and frozen mean RMSE only",
             "frontier":[r["model"] for r in rows if r["pareto_core_flops_RMSE"]]}
    (destination/"pareto_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scientific-source",required=True);p.add_argument("--architecture-inputs",required=True)
    p.add_argument("--source-roots",required=True);p.add_argument("--output",required=True);p.add_argument("--benchmark");a=p.parse_args()
    document=run(a.scientific_source,a.architecture_inputs,a.source_roots,a.output)
    if a.benchmark:write_pareto(document,a.benchmark,a.output)
