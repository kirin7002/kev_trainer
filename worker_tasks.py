"""Fixed CPU-only preparation command. It never imports or calls kev.train."""
import argparse
import json
from pathlib import Path

from datasets import DataError, assert_disjoint, fingerprint
from phase_one import system_one


def preflight(value):
    partitions = value['partitions']
    if set(partitions) != {'train', 'calibration', 'validation'}:
        raise DataError('Worker 输入只允许 train/calibration/validation。')
    for role, rows in partitions.items():
        if fingerprint(rows) != value['partition_sha256'][role]:
            raise DataError('分区内容哈希不匹配。')
        system_one(rows, {'objects': value['objects']}, role)
    for a, b in (('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')):
        assert_disjoint(partitions[a], partitions[b])
    return {'kind': 'cpu_preflight_v1', 'training_verified': False,
            'counts': {role: len(rows) for role, rows in partitions.items()},
            'hashes_verified': True, 'full_object_labels_verified': True,
            'gpu_verified': False, 'token_length_verified': False,
            'training_ready': False, 'blockers': ['kev_runtime_unverified',
                'checkpoint_and_lora_targets_unverified', 'tokenizer_length_unverified', 'gpu_unverified']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = preflight(json.loads(Path(args.input).read_text()))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
