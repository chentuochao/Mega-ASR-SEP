"""Verify threshold selection, audio recovery, and non-destructive exports."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from flag_wer_audio import export_flagged_audio


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def write_wav(path):
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b'\x00\x00\x01\x00' * 80)


class FlaggedAudioTests(unittest.TestCase):
    def fixture(self, root):
        sep, mix, clean = [root / name for name in ['sep.wav', 'mix.wav', 'clean.wav']]
        for path in (sep, mix, clean):
            write_wav(path)
        records = [dict(sample_id=key, reference='spoken question', generated_transcription='wrong question',
                        wer=wer, label_wer=wer, effective_action=action,
                        audio_sep=str(sep), audio_mix=str(mix))
                   for key, wer, action in [('boundary', .75, 'REJECT'),
                                            ('bad-respond', .7501, 'RESPOND'),
                                            ('../worst', 2.5, 'REJECT')]]
        return records, (sep, mix, clean)

    def test_strict_threshold_both_actions_and_verified_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, audio = self.fixture(root)
            before = {p.resolve(): p.read_bytes() for p in audio}
            result = export_flagged_audio(records, root / 'flagged', threshold=.75)
            self.assertEqual(result['n_samples'], 3)
            self.assertEqual(result['n_flagged'], 2)
            self.assertEqual(result['flagged_action_counts'], dict(RESPOND=1, REJECT=1))
            self.assertEqual(result['copied_audio_counts'], dict(separated=2, mixture=2))
            exported = [json.loads(line) for line in (root / 'flagged/manifest.jsonl').read_text().splitlines()]
            self.assertEqual([r['sample_id'] for r in exported], ['../worst', 'bad-respond'])
            for record in exported:
                folder = Path(record['folder'])
                self.assertEqual(folder.parent, (root / 'flagged').resolve())
                self.assertIn('ASR transcript:', (folder / 'transcript.txt').read_text())
                for item in record['copied_audio'].values():
                    self.assertEqual(Path(item['path']).read_bytes(), before[Path(item['source'])])
                    self.assertEqual(item['sha256'], hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
            self.assertEqual(before, {p.resolve(): p.read_bytes() for p in audio})
            with self.assertRaises(FileExistsError):
                export_flagged_audio(records, root / 'flagged', threshold=.75)

    def test_expired_paths_join_by_id_and_clean_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, (sep, mix, clean) = self.fixture(root)
            source = []
            for record in reversed(records):
                source.append(dict(utt_id=record['sample_id'], text=record['reference'], reference='QA answer',
                                   audio_sep=sep.name, audio_mix=mix.name,
                                   libritts_path=clean.name if record['sample_id'] == '../worst' else 'missing.wav'))
                record.update(audio_sep='/scr/expired/sep.wav', audio_mix='/scr/expired/mix.wav')
            manifest = root / 'original.jsonl'
            write_jsonl(manifest, source)
            result = export_flagged_audio(records, root / 'flagged', threshold=.75,
                                          source_manifest=manifest, include_clean_reference=True)
            self.assertEqual(result['n_flagged'], 2)
            self.assertEqual(result['copied_audio_counts']['clean_reference'], 1)
            self.assertEqual(result['missing_clean_reference'], 1)
            self.assertEqual(result['provenance']['source_manifest']['sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_invalid_sources_fail_before_creating_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, audio = self.fixture(root)
            manifest = root / 'source.jsonl'
            wrong = dict(sample_id='../worst', text='different question', audio_sep=str(audio[0]), audio_mix=str(audio[1]))
            write_jsonl(manifest, [wrong, dict(wrong, sample_id='bad-respond')])
            with self.assertRaisesRegex(ValueError, 'transcript mismatch'):
                export_flagged_audio(records, root / 'mismatch', threshold=.75, source_manifest=manifest)
            self.assertFalse((root / 'mismatch').exists())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                export_flagged_audio([records[1], records[1]], root / 'duplicate', threshold=.75)
            self.assertFalse((root / 'duplicate').exists())
            records[1]['audio_mix'] = str(root / 'missing.wav')
            with self.assertRaises(FileNotFoundError):
                export_flagged_audio(records, root / 'missing', threshold=.75)
            self.assertFalse((root / 'missing').exists())

    def test_invalid_wer_and_zero_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, _ = self.fixture(root)
            for threshold in (-1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    export_flagged_audio(records, root / 'invalid', threshold=threshold)
            for wer in (-1, float('nan'), float('inf'), None):
                with self.assertRaises(ValueError):
                    export_flagged_audio([dict(records[0], wer=wer, label_wer=wer)], root / 'invalid', threshold=.75)
            result = export_flagged_audio(records, root / 'empty', threshold=3)
            self.assertEqual(result['n_flagged'], 0)
            self.assertEqual((root / 'empty/manifest.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
