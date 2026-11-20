"""Read-only review of accepted binaries and original metadata; never fits/exports.

The private registry is only a locator. Expected hashes are rechecked against bytes.
"""
from pathlib import Path
import argparse, hashlib, json, platform
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cohort');parser.add_argument('--e2-metadata')
    args=parser.parse_args()
    import onnxruntime as ort
    rows=json.loads(Path(args.registry).read_bytes())['models'];result=[]
    for row in rows:
        if row['profile']!='offline_cpu_export_20261120': continue
        metadata_bytes=Path(row['metadata_path']).read_bytes();meta=json.loads(metadata_bytes);binary=Path(row['model_path']).read_bytes()
        actual_model=hashlib.sha256(binary).hexdigest();actual_metadata=hashlib.sha256(metadata_bytes).hexdigest()
        assert actual_model==row['model_sha256']==meta['model_sha256']
        assert actual_metadata==row['metadata_sha256']
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(binary,sess_options=options,providers=['CPUExecutionProvider'])
        inputs=[{'name':x.name,'type':x.type,'shape':x.shape} for x in session.get_inputs()]
        outputs=[{'name':x.name,'type':x.type,'shape':x.shape} for x in session.get_outputs()]
        assert inputs==[{'name':'windows','type':'tensor(float)','shape':[1,7,5]}]
        assert len(outputs)==1 and outputs[0]['name']=='normalized_prediction'
        assert meta['seed']==271801 and meta['preprocessing']['fit_count']==655
        assert meta['output_scale']==0.020953875259434797
        assert meta['calibration_q']==meta['calibrator_source']['q']
        assert meta['calibrator_source']['n']==53
        assert meta['calibrator_source']['date_end']=='2024-12-30'
        assert meta['scientific_source']['checkpoint_sha256']==meta['calibrator_source']['checkpoint_sha256']
        result.append({'model':meta['model'],'seed':meta['seed'],'model_sha256':actual_model,'metadata_sha256':actual_metadata,'model_bytes':len(binary),'input':inputs,'outputs':outputs,'output_scale':meta['output_scale'],'calibration_q_sigma7':meta['calibration_q'],'calibration_valid_after':meta['calibrator_source']['date_end'],'fit_count':655,'has_ood_in_original_metadata':'ood' in meta,'has_reference_in_original_metadata':'explanation_reference_raw' in meta,'requires_contract_extension':True,'extension_scope':'Add verified fit-only OOD limits and first fit raw reference where absent; do not alter original metadata, preprocessing, weights or q.','status':'passed_binary_identity_and_contract'})
    assert len(result)==9
    output={'schema_version':1,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'onnxruntime':ort.__version__,'models':result,'models_checked':9,'new_model_fits':0,'new_exports':0,'new_calibration_fits':0,'scientific_confirmation':False,'limitations':['Verifies binary identity, shape and metadata binding; functional parity and product tests are separate.','Local private registry paths are locators, not public scientific evidence URLs.','No service acceptance or deployment is asserted.']}
    if bool(args.cohort)!=bool(args.e2_metadata):
        raise ValueError('Supply --cohort and --e2-metadata together')
    if args.cohort:
        import numpy as np
        cohort=Path(args.cohort)
        assert hashlib.sha256(cohort.read_bytes()).hexdigest()=='186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
        with np.load(cohort,allow_pickle=False) as z:
            m=json.loads(Path(args.e2_metadata).read_bytes())
            checks={'cohort_sha256':hashlib.sha256(cohort.read_bytes()).hexdigest(),
                    'scaler_mean_exact':np.array_equal(z['training_channel_mean'],m['preprocessing']['mean']),
                    'scaler_std_exact':np.array_equal(z['training_channel_std'],m['preprocessing']['std']),
                    'minimum_exact':np.array_equal(z['training_X'].min(axis=0),m['ood']['minimum_standardized']),
                    'maximum_exact':np.array_equal(z['training_X'].max(axis=0),m['ood']['maximum_standardized']),
                    'reference_raw_exact':np.array_equal(z['training_raw'][0],m['explanation_reference_raw']),
                    'reference_target':str(z['training_dates'][0]),'fit_count':len(z['training_X'])}
            assert all(v for k,v in checks.items() if k.endswith('_exact'))
        output['shared_E2_fit_descriptor_check']=checks
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'models_checked':len(result),'status':'passed_binary_identity_and_contract','new_model_fits':0}))

if __name__=='__main__':main()
