# SALMONN 批量推理脚本

## 脚本说明

### 1. 多进程批量推理 (推荐)
**脚本**: `inference_batch_multiprocess_single_file.py`

**适用场景**: 多GPU环境，需要快速处理大量数据

**核心参数**:
```bash
python inference_batch_multiprocess_single_file.py \
    --checkpoint_path <模型路径> \
    --write_path_title <输出文件名> \
    --worker_num <GPU数量> \
    --batch_size <批次大小>
```

**注意事项**:
- `worker_num` 建议等于GPU数量，每个worker占用1个GPU
- `batch_size` 过大可能导致OOM，建议从5开始调整
- 输出自动保存到: `/mnt/bn/audio-visual-llm-data6/terumi/SALMONNv1.1_eval/Inference_data/`

---

### 2. 单进程批量推理
**脚本**: `inference_batch_test_set.py`

**适用场景**: 单GPU调试或小规模测试

**核心参数**:
```bash
python inference_batch_test_set.py \
    --checkpoint_path <模型路径> \
    --write_path_title <输出文件名> \
    --batch_size <批次大小>
```

---

## 关键功能

### ✅ 自动模型识别
根据 `checkpoint_path` 自动识别模型类型：
- 包含 `dasheng_wavlm` → 使用 dasheng_wavlm 编码器
- 包含 `whisper_beats` → 使用 whisper_beats 编码器  
- 包含 `qwen` → 使用 qwenomni 编码器
- 其他 → 默认 zipformer2 编码器

### ✅ 任务过滤
只推理特定任务类型：
```bash
--tasks_to_infer ['asr', 'audiocaption', 'translation_ec']
```

**支持的任务**: `audiocaption`, `asr`, `translation_ec`, `phone_recognition`, `speech_query`, `speaker_verification`, `emotion_recognition`, `gender_QA`, `speech_separation`, `slot_filling`, `music_description`

### ✅ 断点续传
```bash
--start <行号>  # 从上次中断的位置继续
```

### ✅ Debug模式
```bash
--debug  # 自动切换为：小数据集、batch_size=2、worker_num=1、输出到logs/debug/
```

---

## 用户必知

### ⚠️ 重要警告

1. **输出文件会追加而非覆盖**
   - 重新运行前请删除旧文件或更换 `write_path_title`

2. **GPU内存管理**
   - 多进程时每个worker独立占用GPU内存
   - 出现OOM时同时减小 `batch_size` 和 `worker_num`

3. **路径规范**
   - `write_path_title` 不能包含下划线 `_`
   - 最终输出文件名自动添加后缀: `_MultP_Batch_AsrAacAstQueryPhoneSvGenderEmoSepKeSfMusic.jsonl`

4. **多进程数据顺序**
   - 多进程推理后自动按原始顺序合并结果
   - 临时文件保存在 `{write_path_title}_temp_dir/`，完成后自动清理

---

## 快速开始示例

### 场景1: 完整推理 (3卡)
```bash
python inference_batch_multiprocess_single_file.py \
    --checkpoint_path /path/to/your/model \
    --write_path_title my_experiment \
    --worker_num 3 \
    --batch_size 5
```

### 场景2: 只推理ASR任务
```bash
python inference_batch_multiprocess_single_file.py \
    --checkpoint_path /path/to/your/model \
    --write_path_title asr_only \
    --tasks_to_infer ['asr'] \
    --worker_num 3 \
    --batch_size 10
```

### 场景3: 调试模式
```bash
python inference_batch_multiprocess_single_file.py \
    --checkpoint_path /path/to/your/model \
    --write_path_title debug_test \
    --debug
```

### 场景4: 断点续传 (从第1000条开始)
```bash
python inference_batch_multiprocess_single_file.py \
    --checkpoint_path /path/to/your/model \
    --write_path_title continue_exp \
    --start 1000 \
    --worker_num 3
```

---

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CUDA out of memory | batch_size或worker_num过大 | 减小batch_size和worker_num |
| 输出结果顺序错乱 | 多进程合并逻辑异常 | 检查temp_dir是否完整，查看日志 |
| 模型加载失败 | checkpoint_path错误或缺少文件 | 确认路径包含config.json和pytorch_model.bin |
| 推理结果为空 | task_filter过滤掉所有数据 | 检查tasks_to_infer参数 |

---

## 输出格式

每行JSON包含:
```json
{
    "path": "音频路径",
    "task": "任务类型", 
    "Q": "问题(如果有)",
    "response": "模型生成的回答"
}
```