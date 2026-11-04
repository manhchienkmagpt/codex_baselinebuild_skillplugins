"""Offline regression checks. Run: python tests/test_validate_artifacts.py"""
from pathlib import Path
import importlib.util
import json
import shutil
import unittest
import uuid

SKILL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', SKILL / 'scripts/validate_artifacts.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
DOMAIN = 'cv' if validator.METADATA_KEY == 'cv_baseline' else 'nlp'


def notebook(variant='hardcore', extra=''):
    declaration = {'variant': variant, 'task_type': 'classification' if DOMAIN == 'cv' else 'text_only',
                   'pretraining_policy': 'scratch', 'pretrained_components': []}
    if variant == 'hardcore':
        declaration.update(training_status='untrained', inference={
            'split': 'test', 'id_field': 'id', 'prediction_field': 'prediction', 'prediction_type': 'integer'})
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {validator.METADATA_KEY: declaration}, 'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': '# Ví dụ\n## 1. Load dữ liệu cơ bản\n## 2. Định nghĩa model và inference'},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': '''import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Linear(1,2)
    def forward(self,raw):
        return self.head(torch.tensor(raw,dtype=torch.float32))
model=Model()
model.eval()
with torch.inference_mode():
    predictions=model([[1.0]]).argmax(-1).tolist()
assert len(predictions)==1
''' + extra}]}


class VariantContracts(unittest.TestCase):
    def setUp(self):
        # Inherit the workspace ACL; Windows sandbox users may not share pytest mode=0700 ACLs.
        self.directory = Path.cwd().resolve() / ('.validator-test-' + uuid.uuid4().hex)
        self.directory.mkdir()

    def tearDown(self):
        target = self.directory.resolve()
        if target.parent != Path.cwd().resolve() or not target.name.startswith('.validator-test-'):
            raise RuntimeError('Unexpected test cleanup target')
        shutil.rmtree(target)

    def scan(self, value):
        path = self.directory / 'notebook.ipynb'
        path.write_text(json.dumps(value), encoding='utf-8')
        return validator.scan_notebook(path)

    def test_inline_and_helper_training_with_inference_checkpoint(self):
        for variant in ("optimized", "basic"):
            for helper in (False, True):
                with self.subTest(variant=variant, helper=helper):
                    value = notebook(variant)
                    training = "for epoch in range(2):\n    model.train()\n    for raw,target in train_loader:\n        optimizer.zero_grad()\n        loss=criterion(model(raw),target)\n        loss.backward()\n        optimizer.step()\n"
                    if helper:
                        training = "def train_phase(model, loader):\n    model.train()\n    for raw,target in loader:\n        optimizer.zero_grad()\n        loss=criterion(model(raw),target)\n        loss.backward()\n        optimizer.step()\nfor epoch in range(2):\n    train_phase(model,train_loader)\n"
                    value["cells"][-1]["source"] += "\n" + training + "torch.save({'model_state':model.state_dict()},'best.pt')\nmodel.load_state_dict(torch.load('best.pt')['model_state'])\n"
                    errors, warnings = self.scan(value)
                    self.assertFalse(errors)
                    self.assertFalse(any("checkpoint" in warning.lower() for warning in warnings))

    def scan_report_text(self, content):
        path = self.directory / "huong_giai_quyet.md"
        path.write_text(content, encoding="utf-8")
        return validator.scan_report(path)

    def report_parts(self):
        bodies = ["Đầu vào là một mẫu; đầu ra là nhãn. Ví dụ minh họa và metric được giải thích.",
                  "Dữ liệu có nhãn và split train/validation/test. So sánh hai hướng dựa trên dữ liệu.",
                  "Chỉ fit biểu diễn trên train; giữ thông tin quan trọng và nhãn đúng.",
                  "Chọn mô hình phù hợp ngân sách; giải thích lợi ích và giới hạn.",
                  "Tham số | Giá trị | Vai trò | Lý do | Tăng/giảm\nC | 1 | điều chuẩn | đề xuất có cơ sở | cần kiểm validation"]
        return [validator.REPORT_HEADINGS[0], *[heading+"\n\n"+body for heading,body in zip(validator.REPORT_HEADINGS[1:],bodies)]]

    def test_five_part_report_valid(self):
        self.assertFalse(self.scan_report_text("\n\n".join(self.report_parts()))[0])

    def test_report_missing_section(self):
        parts = self.report_parts(); parts.pop(2)
        self.assertTrue(self.scan_report_text("\n\n".join(parts))[0])

    def test_report_wrong_order(self):
        parts = self.report_parts(); parts[2],parts[3] = parts[3],parts[2]
        self.assertTrue(self.scan_report_text("\n\n".join(parts))[0])

    def test_report_empty_section(self):
        parts = self.report_parts(); parts[3] = validator.REPORT_HEADINGS[3]+"\n<!-- no explanation -->\n### Chưa có nội dung"
        self.assertTrue(self.scan_report_text("\n\n".join(parts))[0])

    def test_report_duplicate_section(self):
        parts = self.report_parts(); parts.append(parts[2])
        self.assertTrue(self.scan_report_text("\n\n".join(parts))[0])

    def test_report_rejects_legacy_nine_sections(self):
        legacy = "\n\n".join([validator.REPORT_HEADINGS[0], *[f"## {i}. Mục cũ\nNội dung cũ" for i in range(1,10)]])
        self.assertTrue(self.scan_report_text(legacy)[0])

    def test_report_rejects_extra_appendix(self):
        self.assertTrue(self.scan_report_text("\n\n".join(self.report_parts())+"\n\n## Phụ lục\nLịch sử kiểm thử")[0])

    def test_report_ignores_headings_inside_code_example(self):
        parts = self.report_parts(); parts[1] += "\n\n```text\n## Heading in example\n```"
        self.assertFalse(self.scan_report_text("\n\n".join(parts))[0])

    def test_hardcore_needs_no_checkpoint(self):
        errors, warnings = self.scan(notebook())
        self.assertFalse(errors)
        self.assertFalse(any('checkpoint' in item.lower() for item in warnings))

    def test_training_checkpoint_and_loss_rejected(self):
        for operation in ('loss.backward()', 'model.train()', 'features.fit_transform(["text"])',
                          'optimizer=torch.optim.Adam(model.parameters())', 'loss=nn.CrossEntropyLoss()',
                          'torch.save(model.state_dict(),"x.pt")',
                          'from torch.optim import SGD as Learner\noptimizer=Learner(model.parameters(),lr=0.1)',
                          'for epoch in range(2):\n    print(epoch)',
                          'def train_epoch():\n    return 0'):
            with self.subTest(operation=operation):
                self.assertTrue(self.scan(notebook(extra='\n' + operation))[0])

    def test_contract_and_raw_model_call_required(self):
        for change in ('missing_inference', 'train_split', 'trained', 'pretrained', 'no_forward_call', 'external_preprocessing'):
            with self.subTest(change=change):
                value = notebook()
                meta = value['metadata'][validator.METADATA_KEY]
                if change == 'missing_inference':
                    meta.pop('inference')
                elif change == 'train_split':
                    meta['inference']['split'] = 'train'
                elif change == 'trained':
                    meta['training_status'] = 'trained'
                elif change == 'pretrained':
                    meta['pretraining_policy'] = validator.COMPONENT_POLICY
                elif change == 'no_forward_call':
                    value['cells'][-1]['source'] = value['cells'][-1]['source'].replace('model([[1.0]]).argmax(-1).tolist()', '[1]')
                else:
                    value['cells'][-1]['source'] += '\ndef preprocess(raw):\n    return raw.lower()\n'
                self.assertTrue(self.scan(value)[0])

    def test_basic_rejects_optimized_preprocessing(self):
        for extra in ('\nimport sentencepiece as spm', '\nx=TfidfVectorizer()', '\nx=RandomHorizontalFlip()'):
            self.assertTrue(self.scan(notebook('basic', extra))[0])

    def test_basic_cannot_enable_pretrained(self):
        value = notebook('basic')
        value['metadata'][validator.METADATA_KEY].update(pretraining_policy=validator.COMPONENT_POLICY,
            task_type='classification' if DOMAIN == 'cv' else 'image_captioning', pretrained_components=[{
                'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1',
                'role': 'image_encoder', 'input': 'image', 'output': 'features', 'training': 'frozen'}])
        self.assertTrue(any('basic requires scratch' in error for error in self.scan(value)[0]))

    def test_prediction_count_order_types_and_entities(self):
        contract = {'id_field': 'id', 'prediction_field': 'prediction', 'prediction_type': 'integer'}
        self.assertFalse(validator.validate_predictions([{'id': 'a', 'prediction': 1}], ['a'], contract))
        for rows in ([], [{'id': 'a', 'prediction': 'bad'}], [{'id': 'b', 'prediction': 1}], [{'id': 'a'}], [{'id': 'a', 'prediction': True}]):
            self.assertTrue(validator.validate_predictions(rows, ['a'], contract))
        contract['prediction_type'] = 'entities'
        self.assertFalse(validator.validate_predictions([{'id': 'a', 'prediction': [{'start_word': 0, 'end_word': 2, 'type': 'ANSWER'}]}], ['a'], contract))
        self.assertTrue(validator.validate_predictions([{'id': 'a', 'prediction': [{'start_word': 2, 'end_word': 1, 'type': 'ANSWER'}]}], ['a'], contract))

    def test_malformed_metadata_is_diagnostic(self):
        value = notebook()
        value['metadata'] = []
        self.assertTrue(self.scan(value)[0])

    @unittest.skipUnless(DOMAIN == 'cv', 'CV-only backbone policy')
    def test_cv_image_video_task_types(self):
        for task in ('classification', 'image_pair', 'detection', 'semantic_segmentation', 'instance_segmentation',
                     'keypoints', 'restoration', 'video_classification', 'action_recognition', 'tracking'):
            value = notebook('basic')
            value['metadata'][validator.METADATA_KEY]['task_type'] = task
            self.assertFalse(self.scan(value)[0])

    @unittest.skipUnless(DOMAIN == 'cv', 'CV-only backbone policy')
    def test_cv_backbone_allowed_full_detector_forbidden(self):
        value = notebook('optimized')
        value['metadata'][validator.METADATA_KEY].update(pretraining_policy='backbone_components', pretrained_components=[{
            'source': 'torchvision', 'model_id': 'resnet18', 'revision': 'ResNet18_Weights.IMAGENET1K_V1',
            'role': 'image_encoder', 'input': 'RGB', 'output': 'features', 'training': 'finetune'}])
        value['cells'][-1]['source'] = 'from torchvision.models import resnet18,ResNet18_Weights\nbackbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)'
        self.assertFalse(self.scan(value)[0])
        value['cells'][-1]['source'] = 'from torchvision.models.detection import fasterrcnn_resnet50_fpn\nmodel=fasterrcnn_resnet50_fpn(weights="DEFAULT")'
        self.assertTrue(any('full-task' in error for error in self.scan(value)[0]))

    @unittest.skipUnless(DOMAIN == 'cv', 'CV-only backbone policy')
    def test_cv_pretrained_video_backbone(self):
        value = notebook('optimized')
        value['metadata'][validator.METADATA_KEY].update(task_type='video_classification',
            pretraining_policy='backbone_components', pretrained_components=[{
                'source': 'torchvision', 'model_id': 'r3d_18', 'revision': 'R3D_18_Weights.KINETICS400_V1',
                'role': 'video_encoder', 'input': 'video clip', 'output': 'clip features', 'training': 'finetune'}])
        value['cells'][-1]['source'] = 'from torchvision.models.video import r3d_18,R3D_18_Weights\nbackbone=r3d_18(weights=R3D_18_Weights.KINETICS400_V1)'
        self.assertFalse(self.scan(value)[0])

    @unittest.skipUnless(DOMAIN == 'cv', 'CV-only backbone policy')
    def test_cv_scratch_detector_disables_backbone_default(self):
        value = notebook('basic')
        value['cells'][-1]['source'] = 'from torchvision.models.detection import fasterrcnn_resnet50_fpn\nmodel=fasterrcnn_resnet50_fpn(weights=None)'
        self.assertTrue(any('weights_backbone' in error for error in self.scan(value)[0]))
        value['cells'][-1]['source'] = value['cells'][-1]['source'].replace('weights=None', 'weights=None,weights_backbone=None')
        self.assertFalse(self.scan(value)[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
