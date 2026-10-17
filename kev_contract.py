"""Source-verified command and checkpoint inspection support, never an executor.

See kev-source-contract.json and docs/KEV_CONTRACT.md. No torch imports, downloads,
remote shell, model loading or training happen in this module.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from datasets import DataError, parse_json

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / 'kev-source-contract.json').read_text())
EXPECTED_MODEL = next(model for model in json.loads((ROOT / 'models.json').read_text()) if model['id'] == 'kev-0.8b-v1.0')


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    argv: tuple[str, ...]
    output_directory: str


def verify_source_files(root):
    """Check an explicitly supplied dependency checkout without importing it."""
    root = Path(root).resolve()
    for relative, expected in CONTRACT['files'].items():
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']):
            raise DataError(f'固定 Kev 源码文件不匹配：{relative}')
    return {'commit': CONTRACT['commit'], 'files_verified': len(CONTRACT['files']),
            'verification': 'selected_source_files_only', 'runtime_verified': False}


def build_manual_commands(config, workspace, *, python='python', checkpoint_compatibility=None):
    """Build argv for review only. Requires an independently supplied compatibility record.

    The caller still needs runtime/source-tree/weights verification and one shared
    wall-clock budget. Returned commands are never automatically enqueued/run.
    Calibration writes and final test evaluation are deliberately absent.
    """
    model, parameters = config['model'], config['parameters']
    if model.get('id') != 'kev-0.8b-v1.0' or model.get('kev_commit') != CONTRACT['commit']:
        raise DataError('仅支持固定提交的第一阶段 0.8B 模型。')
    if any(model.get(key) != EXPECTED_MODEL[key] for key in ('checkpoint', 'base', 'base_revision', 'lora_rank', 'head_dim')):
        raise DataError('拒绝覆盖固定模型的 checkpoint/base/revision/rank/head。')
    expected = {'base': model['base'], 'base_revision': model['base_revision'], 'lora': model['lora_rank'],
                'head_dim': model['head_dim'], 'weights': 'lora', 'option_isolation': False, 'special_embeddings': False}
    if not isinstance(checkpoint_compatibility, dict) or any(checkpoint_compatibility.get(k) != v for k, v in expected.items()):
        raise DataError('缺少与基座/revision/rank/head 匹配的独立 checkpoint 兼容记录。')
    if checkpoint_compatibility.get('lora_targets') not in ('all', 'dense', 'attn', 'qv'):
        raise DataError('需要从实际 adapter_config/已验证架构确定 lora_targets，不能猜测。')
    allowed = {'learning_rate', 'epochs', 'batch_size', 'gradient_accumulation', 'precision', 'seed'}
    if set(parameters) != allowed or parameters['precision'] not in CONTRACT['supported_training_dtypes']:
        raise DataError('训练参数不在固定 CLI 白名单内。')
    for name, low, high in (('epochs', 1, 20), ('batch_size', 1, 64), ('gradient_accumulation', 1, 256), ('seed', 0, 2**32-1)):
        value = parameters[name]
        if type(value) is not int or not low <= value <= high:
            raise DataError('训练整数参数无效。')
    rate = parameters['learning_rate']
    if type(rate) not in (int, float) or not 1e-7 <= rate <= .001 or not math.isfinite(rate):
        raise DataError('学习率无效。')
    root = Path(workspace).resolve()
    # Fixed names within a fresh attempt: no user-provided executable, cwd,
    # shell fragments, output paths, resume flags, test or regression files.
    for output in ('smoke', 'candidate', 'baseline-validation', 'candidate-calibration', 'candidate-validation'):
        if (root / output).exists():
            raise DataError('拒绝覆盖既有输出目录；重试必须使用新的 attempt 工作区。')
    common = ('--base', model['base'], '--base_revision', model['base_revision'],
              '--init_from', model['checkpoint'], '--lora', str(model['lora_rank']), '--head_dim', str(model['head_dim']),
              '--lora_targets', checkpoint_compatibility['lora_targets'], '--dtype', parameters['precision'],
              '--device', 'cuda', '--checkpointing', '1', '--lr', str(rate), '--batch', str(parameters['batch_size']),
              '--accum', str(parameters['gradient_accumulation']), '--seed', str(parameters['seed']))
    train = (python, '-m', 'kev.train', '--data', str(root / 'train.jsonl'), *common)
    def benchmark(stage, run, role):
        return CommandSpec(stage, (python, '-m', 'kev.benchmark', '--run', str(run), '--data', str(root / (role + '.jsonl')),
                                   '--device', 'cuda', '--out', str(root / stage)), str(root / stage))
    return [benchmark('baseline-validation', model['checkpoint'], 'validation'),
            CommandSpec('smoke', (*train, '--epochs', '1', '--max_steps', '1', '--out', str(root / 'smoke')), str(root / 'smoke')),
            CommandSpec('candidate', (*train, '--epochs', str(parameters['epochs']), '--out', str(root / 'candidate')), str(root / 'candidate')),
            benchmark('candidate-calibration', root / 'candidate', 'calibration'),
            benchmark('candidate-validation', root / 'candidate', 'validation')]


def inspect_checkpoint(root, expected_model):
    """Structural/hash inspection only. Does not deserialize head.pt or claim loadability.

    A LoRA package still requires the separately pinned base. The current source
    loader, tensor compatibility and calibration must be checked in a real model
    environment before treating it as a downloadable validated model.
    """
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise DataError('模型目录不存在或为符号链接。')
    required = ('adapter_config.json', 'adapter_model.safetensors', 'head.pt', 'tokenizer_config.json',
                'training_config.json', 'training_metrics.json')
    if any(not (root / name).is_file() for name in required):
        raise DataError('模型包缺少 adapter/head/tokenizer/训练配置或训练指标。')
    if not (root / 'tokenizer.json').is_file() and not all((root / name).is_file() for name in ('vocab.json', 'merges.txt')):
        raise DataError('模型包缺少完整 tokenizer 数据。')
    files = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise DataError('模型包不允许符号链接。')
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(block)
            files[path.relative_to(root).as_posix()] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    adapter = parse_json((root / 'adapter_config.json').read_bytes())
    training = parse_json((root / 'training_config.json').read_bytes())
    if not isinstance(adapter, dict) or not isinstance(training, dict):
        raise DataError('Adapter/训练配置必须为 JSON 对象。')
    if adapter.get('r') != expected_model['lora_rank'] or adapter.get('base_model_name_or_path') != expected_model['base']:
        raise DataError('Adapter 基座或 rank 与选定模型不兼容。')
    if training.get('base_revision') != expected_model['base_revision']:
        raise DataError('训练记录的 base revision 不匹配。')
    return {'kind': 'checkpoint_structural_manifest', 'files': files, 'base': expected_model['base'],
            'base_revision': expected_model['base_revision'], 'load_verified': False, 'training_verified': False,
            'resume_supported': False, 'required_followup': ['trusted_head_tensor_validation',
                'target_module_and_tensor_compatibility', 'current_inference_loader', 'calibration_provenance']}
