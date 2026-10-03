#!/usr/bin/env bash
# End-to-end Distillation pipeline for Kokoro 7.48M Vietnamese on Kaggle
set -e

DATA_DIR="/kaggle/input/datasets/xkorenguyenthinh/ngochuyen001/content/audio"
META_FILE="$DATA_DIR/metadata.csv"

echo "=== GIAI ĐOẠN 1: Tiền xử lý dữ liệu & Trích xuất Voicepack ==="
python prep_manifest_vi.py \
    --data-dir "$DATA_DIR" \
    --metadata "$META_FILE" \
    --out-dir data/wavs \
    --out-manifest data/vi_manifest.jsonl

python extract_voicepack.py \
    --audio-dir data/wavs \
    --output vi_voice.pt

echo "=== GIAI ĐOẠN 2: Huấn luyện CTC Aligner (40Hz) ==="
python train_aligner.py \
    --manifest data/vi_manifest.jsonl \
    --max-frames 1000 \
    --steps 24000 \
    --out runs/aligner_vi

echo "=== GIAI ĐOẠN 3: Forced Alignment & Gate Kiểm định ==="
# 1. Chạy probe 500 clips để tìm best shift
python align_ctc.py \
    --ckpt runs/aligner_vi/last.pt \
    --manifest data/vi_manifest.jsonl \
    --max-frames 1000 \
    --limit 500 \
    --out data/index_probe.jsonl

SHIFT=$(python pick_shift.py data/index_probe.jsonl config_student_wide.json)
echo "[*] Best Shift measured: $SHIFT"

# 2. Forced alignment toàn bộ dataset với $SHIFT
python align_ctc.py \
    --ckpt runs/aligner_vi/last.pt \
    --manifest data/vi_manifest.jsonl \
    --max-frames 1000 \
    --shift "$SHIFT" \
    --seg-max 800 \
    --seg-min 60 \
    --out data/index_raw.jsonl

# 3. Cắt đoạn đảm bảo không vượt quá giới hạn 512 tokens của PLBERT
python resegment.py data/index_raw.jsonl data/index_ctc.jsonl

# 4. Kiểm định Gate 16
python gate16.py data/index_ctc.jsonl config_student_wide.json 600

echo "=== GIAI ĐOẠN 4: Distill / Finetune Student 7M ==="
# Tự động tải kokoro_en_7m.pth nếu chưa có
if [ ! -f "kokoro_en_7m.pth" ]; then
    wget -q https://huggingface.co/oddadmix/Kokoro-7M-Distill/resolve/main/kokoro_en_7m.pth -O kokoro_en_7m.pth
fi

python train_student.py \
    --steps 40000 \
    --batch 8 \
    --lr 5e-5 \
    --mel-weight 5.0 \
    --sil-weight 100.0 \
    --slm-weight 1.0 \
    --slm-adv-weight 0.2 \
    --init-from kokoro_en_7m.pth \
    --voicepack vi_voice.pt \
    --index data/index_ctc.jsonl \
    --config config_student_wide.json \
    --out runs/kokoro_vi_7m \
    --resume \
    --ckpt-every 2000

echo "=== GIAI ĐOẠN 5: Đóng gói Nested Checkpoint & Kiểm thử ==="
python export_model.py \
    --src runs/kokoro_vi_7m/student.pth \
    --dst kokoro_vi_7m.pth

python infer_test.py \
    --text "Xin chào, đây là âm thanh tiếng Việt được tổng hợp thành công từ mô hình cô cô rô bảy mê ga byte." \
    --model kokoro_vi_7m.pth \
    --voicepack vi_voice.pt \
    --out test_sample.wav

echo "=== HOÀN TẤT TOÀN BỘ QUY TRÌNH ==="
