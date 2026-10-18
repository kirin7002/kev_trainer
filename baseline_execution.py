"""Experiment-scoped baseline-only evaluation: no Trial, optimizer or final test."""
import json
from pathlib import Path
import shutil

from datasets import DataError,fingerprint
from model_artifacts import build_bundle,canonical_sha,file_sha
from phase_one import system_one
from pipeline_validation import convert_benchmark,inference_summary
from trial_execution import TrialPipeline,read,write
from kev_contract import CONTRACT

BASELINE_STAGES = ('BASELINE_VALIDATING','BASELINE_RAW','BASELINE_CALIBRATION','BASELINE_CALIBRATING',
                   'BASELINE_SYSTEM','BASELINE_LOADING','BASELINE_COMPARING')


class BaselinePipeline(TrialPipeline):
    def stage(self,name,operation,payload,**extra):
        if name not in BASELINE_STAGES or operation not in ('validate-baseline','benchmark','calibrate','load'):
            raise DataError('baseline-only 作业禁止训练、候选恢复和最终 test。')
        return super().stage(name,operation,payload,**extra)

    def run(self,inputs,job):
        work,config = self.directory,inputs['config']
        if set(inputs['partitions']) != {'calibration','validation'} or config.get('purpose') != 'baseline_only':
            raise DataError('基线作业只接收 calibration/validation。')
        if canonical_sha(config) != inputs['config_sha256']: raise DataError('基线配置 SHA 不匹配。')
        if self.backend.mode=='cuda': self.backend.verify_platform(config)
        for role,rows in inputs['partitions'].items():
            if fingerprint(rows)!=inputs['partition_sha256'][role]: raise DataError('基线输入哈希不匹配。')
            path=work/'data'/(role+'.jsonl');path.parent.mkdir(exist_ok=True)
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in system_one(
                rows,{'objects':inputs['objects'], **config.get('task_policy', {})},role)))
        payload={'work':str(work.resolve()),'mode':'baseline','config':config,'objects':inputs['objects']}
        validation=self.stage('BASELINE_VALIDATING','validate-baseline',payload)
        runtime=validation['runtime'];real=self.backend.mode=='cuda'
        if (self.backend.mode not in ('cuda','software_contract_only') or not validation.get('tokenizer_verified')
                or validation.get('counts')!={role:len(rows) for role,rows in inputs['partitions'].items()}
                or validation.get('rejected_records')!=0 or validation.get('truncated_records')!=0
                or runtime.get('source_commit')!=CONTRACT['commit']
                or runtime.get('parent_archive_sha256')!=CONTRACT['released_checkpoint']['sha256']):
            raise DataError('基线 runtime/tokenizer/来源检查失败。')
        if real and (runtime.get('execution_backend')!='cuda' or runtime.get('runtime_verified') is not True):
            raise DataError('基线 CUDA 未验证。')
        objects=[o['id'] for o in inputs['objects']]
        def benchmark(stage,role,raw):
            receipt=self.stage(stage,'benchmark',payload,model_role='baseline',data_role=role,raw=raw)
            directory=work/receipt['directory']
            if directory.resolve().parent!=(work/'benchmarks').resolve(): raise DataError('基线输出路径越界。')
            records=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines() if line.strip()]
            value=convert_benchmark(records,read(directory/'report.json'),inputs['partitions'][role],objects,inputs['partition_sha256'][role])
            write(work/'reports'/(stage+'-predictions.json'),value)
            return value
        raw=benchmark('BASELINE_RAW','validation',True)
        benchmark('BASELINE_CALIBRATION','calibration',True)
        policy=self.stage('BASELINE_CALIBRATING','calibrate',payload,model_role='baseline')
        self.check_calibration(policy)
        served=benchmark('BASELINE_SYSTEM','validation',False)
        loader=self.stage('BASELINE_LOADING','load',payload)
        if (not loader.get('loader_verified') or loader.get('objects_verified')!=len(objects)
                or loader.get('head_sha256')!=file_sha(work/'baseline/head.pt')
                or loader.get('weights_sha256')!=file_sha(work/'baseline/adapter_model.safetensors')):
            raise DataError('基线当前加载器回执不匹配。')
        self.supervisor.check('BASELINE_COMPARING');self.completed.append('BASELINE_COMPARING')
        write(work/'reports/BASELINE_COMPARING.json',{'source':'authenticated_baseline_worker' if real else 'software_contract_fixture',
            'baseline_verified':real,'training_verified':False,'optimizer_steps':0,'runtime':runtime,
            'validation_dataset_sha256':inputs['partition_sha256']['validation'],
            'inference':inference_summary(served,inputs['partitions']['validation'])})
        if tuple(self.completed)!=BASELINE_STAGES: raise DataError('基线阶段不完整。')
        package=work/'package';package.mkdir()
        for name in ('reports','policy','logs'): shutil.copytree(work/name,package/name)
        (package/'checkpoint').mkdir()
        for name in ('head.pt','adapter_config.json','adapter_model.safetensors'):
            shutil.copyfile(work/'baseline'/name,package/'checkpoint'/name)
        for directory in (work/'benchmarks').iterdir(): shutil.copytree(directory,package/'reports'/('benchmark-'+directory.name))
        write(package/'policy/config.json',config)
        write(package/'policy/source-contract.json',CONTRACT)
        write(package/'policy/data-registry.json',{'partition_sha256':inputs['partition_sha256'],
            'excluded_partitions':['train','test','regression']})
        bundle=work/'baseline.zip'
        build_bundle(package,bundle,{'kind':'kev_baseline_bundle','job_id':job['id'],'trial_id':None,
            'experiment_id':job['experiment_id'],'attempt_id':job['attempt_id'],'config_sha256':inputs['config_sha256'],
            'execution_backend':self.backend.mode,'training_verified':False,'baseline_verified':real,
            'stages':self.completed,'optimizer_steps':0,'resume_supported':False,
            'image_provenance':runtime.get('image_provenance')})
        self.supervisor.check('BASELINE_COMPARING')
        return {'kind':'gpu_baseline_v1','artifact_sha256':file_sha(bundle),'artifact_path':str(bundle),
                'execution_backend':self.backend.mode,'training_verified':False,'baseline_verified':real}
