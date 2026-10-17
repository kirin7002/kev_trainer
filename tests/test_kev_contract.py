"""Source-contract support, synthetic checkpoint files only; no torch/model loading."""
import json
from pathlib import Path

import pytest

from datasets import DataError
from kev_contract import CONTRACT, EXPECTED_MODEL, build_manual_commands, inspect_checkpoint, verify_source_files


def config():
    return {'model': dict(EXPECTED_MODEL), 'parameters': {'learning_rate': .0001, 'epochs': 2,
        'batch_size': 1, 'gradient_accumulation': 1, 'seed': 42, 'precision': 'bf16'}}


def compatibility():
    return {'base': EXPECTED_MODEL['base'], 'base_revision': EXPECTED_MODEL['base_revision'], 'lora': 16,
            'head_dim': 256, 'weights': 'lora', 'option_isolation': False, 'special_embeddings': False, 'lora_targets': 'all'}


def test_review_commands_use_only_verified_flags_and_training_inputs(tmp_path):
    commands = build_manual_commands(config(), tmp_path, checkpoint_compatibility=compatibility())
    assert [item.stage for item in commands] == ['baseline-validation', 'smoke', 'candidate', 'candidate-calibration', 'candidate-validation']
    for command in commands:
        module = command.argv[2].split('.')[-1]
        assert all(arg in CONTRACT[module + '_flags'] for arg in command.argv if arg.startswith('--'))
        assert '--resume' not in command.argv and '--allow-test' not in command.argv
        assert not any(part.endswith('/test.jsonl') or part.endswith('/regression.jsonl') for part in command.argv)
        assert '--device' in command.argv and command.argv[command.argv.index('--device')+1] == 'cuda'
    smoke = commands[1].argv
    assert smoke[smoke.index('--init_from')+1] == EXPECTED_MODEL['checkpoint']
    assert smoke[smoke.index('--max_steps')+1] == '1'
    assert not list(tmp_path.iterdir()), 'Building a review plan must not run or create a model'


@pytest.mark.parametrize('change', ['precision', 'rank', 'revision', 'checkpoint', 'resume', 'compatibility'])
def test_command_builder_rejects_unsupported_or_incompatible_overrides(tmp_path, change):
    cfg, proof = config(), compatibility()
    if change == 'precision': cfg['parameters']['precision'] = 'fp16'
    if change == 'rank': cfg['model']['lora_rank'] = 32
    if change == 'revision': cfg['model']['base_revision'] = 'other'
    if change == 'checkpoint': cfg['model']['checkpoint'] = '--unsafe'
    if change == 'resume': cfg['parameters']['resume'] = True
    if change == 'compatibility': proof = None
    with pytest.raises(DataError):
        build_manual_commands(cfg, tmp_path, checkpoint_compatibility=proof)


def test_attempt_outputs_never_overwrite_existing_checkpoint(tmp_path):
    (tmp_path / 'candidate').mkdir()
    with pytest.raises(DataError):
        build_manual_commands(config(), tmp_path, checkpoint_compatibility=compatibility())


def synthetic_checkpoint(directory):
    # Arbitrary bytes deliberately demonstrate that structural inspection
    # cannot certify tensor validity, model quality or current loader support.
    for name in ('adapter_model.safetensors', 'head.pt', 'tokenizer_config.json', 'tokenizer.json', 'training_metrics.json'):
        (directory / name).write_bytes(b'not-a-real-model-fixture')
    (directory / 'adapter_config.json').write_text(json.dumps({'r': 16, 'base_model_name_or_path': EXPECTED_MODEL['base']}))
    (directory / 'training_config.json').write_text(json.dumps({'base_revision': EXPECTED_MODEL['base_revision']}))


def test_structural_checkpoint_manifest_never_claims_load_or_resume_verified(tmp_path):
    synthetic_checkpoint(tmp_path)
    manifest = inspect_checkpoint(tmp_path, EXPECTED_MODEL)
    assert manifest['load_verified'] is manifest['training_verified'] is manifest['resume_supported'] is False
    assert manifest['files']['head.pt']['bytes'] > 0
    (tmp_path / 'head.pt').unlink()
    with pytest.raises(DataError):
        inspect_checkpoint(tmp_path, EXPECTED_MODEL)


def test_checkpoint_rejects_symlinks_and_mismatched_rank(tmp_path):
    synthetic_checkpoint(tmp_path)
    (tmp_path / 'unexpected-link').symlink_to(tmp_path / 'head.pt')
    with pytest.raises(DataError):
        inspect_checkpoint(tmp_path, EXPECTED_MODEL)
    (tmp_path / 'unexpected-link').unlink()
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 32, 'base_model_name_or_path': EXPECTED_MODEL['base']}))
    with pytest.raises(DataError):
        inspect_checkpoint(tmp_path, EXPECTED_MODEL)


def test_source_verification_refuses_incomplete_or_different_checkout(tmp_path):
    (tmp_path / 'kev').mkdir()
    (tmp_path / 'kev/train.py').write_text('print("not the pinned code")')
    with pytest.raises(DataError):
        verify_source_files(tmp_path)
