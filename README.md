# Kokoro-7M-NgocHuyen: Distill Kokoro 7.48M cho Tiếng Việt (Kaggle)

Thư mục này chứa toàn bộ mã nguồn, cấu hình và Notebook để thực hiện huấn luyện chưng cất (Knowledge Distillation) mô hình **Kokoro-7M (7.48M tham số)** cho tiếng Việt với dataset `ngochuyen001` trên môi trường **Kaggle GPU (Tesla T4 / P100)**.

---

## 1. Cấu trúc thư mục

| Tệp / Thư mục | Mô tả vai trò |
|---|---|
| `train_kokoro_7m_ngochuyen.ipynb` | **Kaggle Notebook chính** chia theo từng cell cho 5 giai đoạn |
| `prep_manifest_vi.py` | Giai đoạn 1: Chuẩn hóa 24kHz FLAC, cắt khoảng lặng, G2P tiếng Việt sang `vi_manifest.jsonl` |
| `extract_voicepack.py` | Giai đoạn 1: Trích xuất style vector 256 chiều (`vi_voice.pt`) |
| `train_aligner.py` | Giai đoạn 2: Huấn luyện BiLSTM CTC Aligner trên lưới 40 Hz |
| `pick_shift.py` | Giai đoạn 3: Tính toán độ dịch khung (`SHIFT`) tối ưu |
| `align_ctc.py` | Giai đoạn 3: Viterbi forced alignment sinh per-phoneme durations |
| `resegment.py` | Giai đoạn 3: Chia nhỏ đoạn audio tránh vượt trần 512 token của ALBERT |
| `gate16.py` | Giai đoạn 3: **Quality Gate 16** kiểm định chất lượng căn chỉnh thời lượng |
| `train_student.py` | Giai đoạn 4: Distill / Finetune mô hình Student 7M với MPD/MSD GAN, WavLM SLM, Mel Loss |
| `export_model.py` | Giai đoạn 5: Đóng gói checkpoint phẳng sang dạng lồng (Nested) cho KModel |
| `infer_test.py` | Giai đoạn 5: Thử nghiệm tổng hợp giọng đọc tiếng Việt từ văn bản |
| `run_pipeline.sh` | Bash script chạy tự động toàn bộ 5 giai đoạn liên tục |
| `config_student_wide.json` | Cấu hình kiến trúc 7.48M (PLBERT 192/3 heads, Decoder 128 channels, 178 tokens) |
| `kokoro_symbols.py` | Bảng ánh xạ 178 ký hiệu âm vị chuẩn của Kokoro |
| `styletts2_gan/` | Bộ discriminator (MPD, MSD, WavLM) và các hàm loss StyleTTS2 |
| `kokoro_patched/` | Runtime patched của Kokoro hỗ trợ nén channels kích thước nhỏ |

---

## 2. Hướng dẫn sử dụng trên Kaggle

### Cách 1: Chạy trực tiếp qua Kaggle Notebook
1. Mở Kaggle, tạo một Notebook mới.
2. Bật **GPU T4 x 1** hoặc **P100** và bật **Internet: ON**.
3. Thêm Dataset input:
   - Thư mục dữ liệu: `/kaggle/input/datasets/xkorenguyenthinh/ngochuyen001/content/audio`
   - Dataset chứa folder `Kokoro-7M-NgocHuyen` (hoặc upload folder này lên).
4. Mở file [train_kokoro_7m_ngochuyen.ipynb](train_kokoro_7m_ngochuyen.ipynb) và chạy tuần tự các cell.

### Cách 2: Chạy qua Command Line (hoặc Kaggle Terminal)
```bash
bash run_pipeline.sh
```

---

## 3. Các điểm mấu chốt kỹ thuật

1. **Khởi tạo trọng số (Warm-start)**:
   - Script tự động tải `kokoro_en_7m.pth` từ Hugging Face để khởi tạo biểu diễn decoder và duration predictor, giúp tiết kiệm 70% thời gian train so với train từ đầu.
2. **CTC Aligner 40 Hz & Gate 16**:
   - Sử dụng CTC aligner căn chỉnh trên chính giọng thật của Ngọc Huyền, đảm bảo `durations` không bị trễ pha như phương pháp DTW.
   - Bắt buộc kiểm tra `gate16.py` đạt đỉnh tại `+0` và True labels gap > +4.0 dB trước khi train student.
3. **Loss phối hợp cho âm thanh sắc nét**:
   - `mel-weight (5.0)`: Tăng cường chi tiết dải tần 1-4 kHz cho phụ âm.
   - `sil-weight (100.0)`: Triệt tiêu tiếng xì nền ở khoảng lặng.
   - `slm-weight & slm-adv-weight`: Tái tạo độ tự nhiên và ngữ điệu qua WavLM.
