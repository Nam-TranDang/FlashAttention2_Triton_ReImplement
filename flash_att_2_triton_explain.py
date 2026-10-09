import os
import sys
import torch

assert torch.cuda.is_available(), "CUDA GPU is required to run this Triton kernel!"
DEVICE = "cuda"

import triton
import triton.language as tl

'''
LƯU Ý: CHỈ ĐỌC CODE NÀY - KHI HIỂU FLASH ATTENTION VÀ KIẾN THỨC VỀ GPU

HIỂU ĐƠN GIẢN CODE:
1. dòng "kernel = flash_attn_2_fwd_kernel[grid](...) - là CPU định nghĩa Fused Kernel Flash Attention 2 sẽ chạy trên GPU và Grid sẽ là số lượng Programs để tính trên GPU.

2. [grid] - định nghĩa số lượng programs tính song song -> một khi chạy flash_attn_2_fwd_kernel[grid] sẽ tạo 8 programs chạy song song hàm/Kernel này

3. Khi chạy mối Program - Triton sẽ GPU gán sẵn tọa độ cho mỗi program, và tl.program_id chỉ đọc lại tọa độ đó để tính toán.
    grid = (triton.cdiv(N_CTX, block_m), H, Z) = (4, 2, 1) # lần lượt 128 context / 32 block = 4; 2 head attention, 1 batch
    
    pid_m = tl.program_id(0)     # 0,1,2,3 (tương ứng với programs)
    pid_h = tl.program_id(1)     # 0,1
    pid_z = tl.program_id(2)     # 0

4. Flash Attention 2 sẽ load Q nằm ở On-chip - dùng để traverse KV
    q = tl.load

5. Cách hoạt động bên trong: 
    QKV tất cả sẽ có Tensor [1, 2, 128, 32] = 8192  phần tử (batch, head, context_length, model_dim)
    Mục tiêu cho QKV tính nhanh - và ta load Q trước:

    * Q sẽ tách 2 head 
        Head 0 - tính từ phần tử 0 - 4095
        Head 1 - tính từ 4096 - 8192

    * Trong mỗi Head (xử lý 4096 phần từ Q = 128 * 32) => tương ứng mỗi head, sẽ Q chia làm 4 lần tile 
    giao cho 4 program tính song song (do context / block M = 128 / 32 = 4)
        Tile Q (0) đi từ 0 - 31
        Tile Q (1) đi từ 32 - 63
        Tile Q (2) đi từ 64 - 95
        Tile Q (3) đi từ 96 - 127

    * Với mối KV sẽ có Tensor [1, 2, 128, 32] = 8192  phần tử, với mối Tile Q program - ta sẽ lần lượt dượt 4 lần của KV như vậy:

    * Tổng cộng nếu tính toán tổng program sẽ là 
        2 (lần tách head của Q) * 4 (mỗi lần tile Q của từng head) = 8 programs 

    * out_ptrs 
'''





# =====================================================================
# CHÚ GIẢI KÝ HIỆU VỊ TRÍ DỮ LIỆU (dùng xuyên suốt file)
# ---------------------------------------------------------------------
#   [HBM]         : VRAM, lớn nhưng CHẬM. Q, K, V, O gốc nằm ở đây.
#   [L2/L1]       : cache trung gian, phần cứng tự quản lý (ta không điều khiển Triton).
#   [THANH GHI]   : register, riêng tư của từng thread, NHANH nhất.
#                   Biến Triton (q, k, v, qk, p, acc, m_i, l_i) sống ở đây.
#   [SHARED MEM]  : SRAM do ta/compiler quản lý, chia sẻ giữa các thread trong 1 block.
#                   Trong Triton KHÔNG có dòng code nào viết trực tiếp vào shared memory;
#                   compiler tự chèn khi cần (chủ yếu quanh tl.dot).
#   [TENSOR CORE] : đơn vị nhân ma trận nhỏ, chạy lệnh mma. Được gọi bởi tl.dot.
#   [CUDA CORE]   : ALU thường, chạy exp / max / sum / cộng / nhân từng phần tử.
#
# QUY TẮC VÀNG CỦA TRITON:
#   - CUDA: bạn viết code cho 1 THREAD.
#   - Triton: bạn viết code cho 1 PROGRAM (≈ 1 Block, chạy trên 1 SM) và thao tác trên
#     CẢ TILE (mảng 2D) cùng lúc. Compiler tự chia tile cho các warp/thread.
# =====================================================================


# =====================================================================
# 1. FLASH ATTENTION 2 FORWARD KERNEL (32x32 TILES WITH TENSOR CORES)
# =====================================================================
# "Fused kernel" = TOÀN BỘ chuỗi: QK^T -> scale -> mask -> softmax -> PV -> normalize
# nằm trong DUY NHẤT hàm này. Nhờ vậy các ma trận trung gian N×N (S, P) KHÔNG bao giờ
# được ghi ra HBM, chúng chỉ là biến tạm trong thanh ghi.
# HBM chỉ bị chạm: ĐỌC Q, K, V  và  GHI O (đúng 1 lần cuối).
# ---------------------------------------------------------------------
# BẢN ĐỒ INPUT -> PROGRAM -> TILE (ví dụ cụ thể của file này)
# ---------------------------------------------------------------------
# Q, K, V, O đều có shape [Z=1 batch, H=2 head, N_CTX=128 token, HEAD_DIM=32 feature], kiểu fp16,
# nằm trong HBM. Mỗi cặp (batch, head) là 1 ma trận 128×32 ĐỘC LẬP: attention chạy riêng cho từng cặp.

#   Một (batch, head):          Q [128×32]                 K, V [128×32]
#                          ┌───────────────┐          ┌───────────────┐
#   pid_m=0 (hàng 0..31)   │  Q tile 0     │          │  K/V tile 0   │ <- vòng lặp 0 (start_n=0)
#   pid_m=1 (hàng 32..63)  │  Q tile 1     │          │  K/V tile 1   │ <- vòng lặp 1 (start_n=32)
#   pid_m=2 (hàng 64..95)  │  Q tile 2     │          │  K/V tile 2   │ <- vòng lặp 2 (start_n=64)
#   pid_m=3 (hàng 96..127) │  Q tile 3     │          │  K/V tile 3   │ <- vòng lặp 3 (start_n=96)
#                          └───────────────┘          └───────────────┘
#   "TILE" = một mảnh cắt theo chiều TOKEN (32 trong 128 token), lấy đủ 32 feature => tile 32×32.


# Grid ("128/32" = 4, 2, 1) = 8 program. Mỗi program (pid_m, pid_h, pid_z) làm:
#   - Đọc 1 Q tile của head pid_h (đọc 1 lần, giữ suốt kernel)
#   - Duyệt qua CẢ 4 K/V tile của CÙNG head đó (vòng for), tích lũy vào acc
#   - Ghi 1 O tile (32×32) ra HBM
# Lưu ý: 4 program cùng 1 head (pid_m = 0..3) đều đọc lại cùng K, V của head đó.
# Lần đọc lặp lại thường trúng L2 cache -> nên không tốn thêm băng thông HBM tương ứng.
# ---------------------------------------------------------------------


@triton.jit     # Compile this Python function into a GPU kernel
def flash_attn_2_fwd_kernel(
    # --- Các tham số dưới đây là CON TRỎ (địa chỉ) tới dữ liệu trong HBM, chưa phải dữ liệu ---
    # Q_ptr: địa chỉ - pointer - con trỏ dữ liệu HBM đầu tiên của tensor Q [Z,H,N_CTX,HEAD_DIM] (Triton lấy từ q.data_ptr()).
    Q_ptr, K_ptr, V_ptr, O_ptr,

    # --- stride: logic chuyển index -> địa chỉ HBM ---
    #   stride_qz = 8192 (total phân từ của Q = 1*2*128*32)
    #   stride_qh = 4096 (total của input đầu vào = 128 * 32)
    #   stride_qm = 32   (= model_dim)
    #   stride_qk = 1

    
    # Địa chỉ Q[z,h,m,k] = Q_ptr + z*stride_qz + h*stride_qh + m*stride_qm + k*stride_qk
    stride_qz, stride_qh, stride_qm, stride_qk,
    
    # Stride của K, cùng ý nghĩa: z=batch, h=head, n=token của K (trục sequence), k=feature.
    # Tách riêng stride cho K vì K có thể có layout khác Q (vd tensor đã transpose/slice), không giả định giống Q.
    stride_kz, stride_kh, stride_kn, stride_kk,
    # Stride của V (tương tự K).
    stride_vz, stride_vh, stride_vn, stride_vk,

    # Stride của O (đầu ra). Dùng để tính địa chỉ GHI (tl.store) ở cuối kernel.
    stride_oz, stride_oh, stride_om, stride_ok,
    # (program_id) đã cho biết "tôi là batch/head nào"; Z, H chỉ được dùng ở Python wrapper để dựng grid.
    Z, H, N_CTX,       # số batch, số head, độ dài sequence (số token) - là số nguyên thường'

    scale,             # 1/sqrt(HEAD_DIM), hệ số chia điểm attention

    # constexpr = hằng số biết tại thời điểm COMPILE. Triton cần nó để chọn layout
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    '''
    Mỗi input sẽ có shape [1, 2, 128, 32]:
        z	chiều 0	    batch	1	(số câu)
        h	chiều 1	    head	2	
        m	chiều 2	    token	128	(hàng của ma trận)
        k	chiều 3	    feature	32	(cột của ma trận)
    
    Còn mỗi Stride để dịch chuyển - logic "con trỏ" chuyển địa chỉ trên HBM: 
        Shape: [1, 2, 128, 32]

        stride_k = 1                    ← sang cột kế tiếp: nhảy 1 phần tử (batch xử lý = 1)
        stride_m = 32                   ← sang hàng kế tiếp: phải bỏ qua 1 hàng đầy = 32 phần tử
        stride_h = 128 × 32  = 4096     ← sang head kế tiếp: bỏ qua 1 head đầy = 128 hàng × 32
        stride_z = 2 × 128 × 32 = 8192  ← sang batch kế tiếp: bỏ qua 1 batch đầy = 2 head × 4096
    '''

    # -----------------------------------------------------------------
    # PHÂN CÔNG VIỆC: mỗi PROGRAM (≈ 1 Block, chạy trên 1 SM) tự hỏi "tôi là ai?"
    # Grid ở wrapper = (4, 2, 1) = 8 program chạy gần như song song trên các SM.
    # Mỗi program chịu trách nhiệm: 1 tile Q của 1 head của 1 batch.
    # -----------------------------------------------------------------
    ''' Đây là 1 tiles - nhận từ Grid cấu hình - grid = (triton.cdiv(N_CTX, block_m), H, Z)'''
    pid_m = tl.program_id(0)     # which Q block (0..3) - Input  | trục 0: khối Q thứ mấy theo chiều sequence
    pid_h = tl.program_id(1)     # trục 1: head thứ mấy
    pid_z = tl.program_id(2)     # trục 2: batch thứ mấy
    # (Đây là các số nguyên scalar, nằm trong thanh ghi. Không có dữ liệu tensor nào di chuyển ở đây.)

    # -----------------------------------------------------------------
    # ĐỊA CHỈ CƠ SỞ: nhảy tới đầu của (batch, head) mà program này phụ trách.
    # Tensor [Batch, Head, N, D] nằm "phẳng" trong HBM; stride cho biết cách nhảy.
    # Ví dụ tensor [1,2,128,32]: stride_h = 128*32 = 4096 -> head 1 bắt đầu cách head 0 4096 phần tử.
    # -----------------------------------------------------------------
    # Base pointers for this batch & head
    q_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    # -----------------------------------------------------------------
    # DỰNG "BẢN ĐỒ CHỈ SỐ" cho tile. tl.arange tạo vector chỉ số [0..N-1].
    # Chưa có dữ liệu nào được đọc; đây chỉ là vector số nguyên nằm trong thanh ghi.
    # -----------------------------------------------------------------
    # How a tile is loaded - Row and dimension offsets inside this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # Vector of indices [0..N-1], which is how you build a tile = 32 rows  [32]
                                                            # = chỉ số HÀNG (token) TUYỆT ĐỐI của khối Q này, vd program 1: 32..63
    offs_n = tl.arange(0, BLOCK_N)                          # chỉ số cột tương đối trong 1 tile K/V: 0..31 (sẽ cộng start_n ở vòng lặp)
    offs_d = tl.arange(0, HEAD_DIM)                        # = 32 columns of features [32]  | chỉ số CỘT feature 0..31

    # -----------------------------------------------------------------
    # Step 1: HBM -> THANH GHI: Load Q block ONE 1 LẦN và giữ suốt kernel
    # -----------------------------------------------------------------
    '''
    offs_m[:, None] is a column [32,1], offs_d[None, :] is a row [1,32]. Broadcasting gives a [32,32] grid of addresses.
    Pointer = base + row × stride_row + col × stride_col.
    One tl.load pulls the whole 32×32 tile. There is no per-thread indexing.
    '''

    # q_ptrs = LƯỚI 32×32 ĐỊA CHỈ (broadcast cột [32,1] với hàng [1,32]).
    # Đây chỉ là "tờ giấy ghi địa chỉ", CHƯA có byte dữ liệu nào di chuyển.
    q_ptrs = Q_ptr + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # q_mask = True ở hàng hợp lệ (< N_CTX). Bảo vệ truy cập ngoài biên khi N_CTX không chia hết 32.
    q_mask = offs_m[:, None] < N_CTX

    '''
    # HBM → on-chip (registers/SRAM) - Mask = Guards out-of-bounds (padding) -
    # ĐÂY LÀ LỆNH THỰC SỰ DI CHUYỂN DỮ LIỆU:
    #   [HBM] --(ld.global, đi qua L2/L1)--> [THANH GHI]

    #   Triton tự chia 1024 phần tử (32×32 fp16) cho các thread; mỗi thread đọc theo cụm

    #   byte liền nhau (coalesced) để tận dụng băng thông. KHÔNG cần threadIdx.
    
    #   Kết quả q là tensor fp16 nằm trong thanh ghi, mỗi thread giữ một mảnh nhỏ của nó.
    #   Chỗ bị mask nhận giá trị other=0.0 (padding).
    '''
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # -----------------------------------------------------------------
    # KHỞI TẠO LOGIC CỦA "ONLINE SOFTMAX"
    # Các biến "online softmax". Tất cả nằm trong [THANH GHI], kiểu fp32 cho chính xác. 
    # Chúng thay thế cho việc phải lưu cả ma trận S (N×N) trong HBM.
    # -----------------------------------------------------------------
    # Running online-softmax variables (kept in registers/SRAM in fp32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # max score seen so far in this tile of Q - running max       (per row)
                                                               # [32]: max điểm đã thấy của mỗi hàng, khởi đầu -inf

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)               # sum of exp(score-m) for normalization - running sum of exp (per row)
                                                               # [32]: tổng exp đã thấy của mỗi hàng (mẫu số softmax)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)     # running output accumulator - running output
                                                               # [32,32]: tích lũy kết quả P@V, chưa chuẩn hóa

    # -----------------------------------------------------------------
    # Step 2: Vòng lặp trong: "STREAM" từng tile K, V dọc chiều sequence - FLASH ATTENTION 2 LOOP
    # Q (đã nằm trên chip) cố định; K, V được kéo từ HBM từng mảnh 32 hàng một.
    # -----------------------------------------------------------------
    '''
    In the inner loop (over j=0,1,2,3), for each tile of K and V:

        1. Load K[tile] and V[tile] into registers/SRAM (not HBM).
        2. Compute QK^T and PV using Tensor Cores (tl.dot).
        3. Update the online-softmax running stats m_i, l_i (in fp32).
        4. Re-scale the accumulated result by the correction factor α = exp(m_i_old - m_i_new).
        5. Write the normalized output tile into HBM at the end.

        Q block (loaded ONCE, stays on-chip)
    │
    ├── iter 0: load K0,V0 → S=QK0ᵀ → softmax update → acc += P·V0
    ├── iter 1: load K1,V1 → ...
    ├── iter 2: load K2,V2 → ...
    └── iter 3: load K3,V3 → ...
            Normalize once → store O to HBM
    '''

    for start_n in range(0, N_CTX, BLOCK_N):    # 0, 32, 64, 96  → 4 iterations
        # Chỉ số hàng TUYỆT ĐỐI của tile K/V ở vòng này (vd vòng 1: 32..63)
        curr_n = start_n + offs_n
        kv_mask = curr_n[:, None] < N_CTX       # mask chống đọc ngoài biên

        # --- Dựng lưới địa chỉ cho tile K và V (vẫn chỉ là "tờ giấy địa chỉ") ---
        # HBM -> THANH GHI: Load current K and V tiles
        k_ptrs = K_ptr + k_offset + curr_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V_ptr + v_offset + curr_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        # --- DỮ LIỆU DI CHUYỂN: [HBM] -> [THANH GHI] (giống q, nhưng lặp lại mỗi vòng) ---
        # k, v bị ghi đè ở vòng sau => thanh ghi được tái sử dụng, không cần chỗ chứa cho cả K, V.
        # (Đây không phải "HBM -> Shared Memory" trực tiếp: tl.load đích đến là thanh ghi.)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)                                           # K tile [32,32] from HBM
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)                                           # V tile [32,32] from HBM

        # -------------------------------------------------------------
        # Matmul #1: S = Q @ K^T   -> chạy trên [TENSOR CORE]
        # tl.dot nhận q, k (fp16, trong thanh ghi) và trả về kết quả fp32 trong thanh ghi.
        # Bên trong (do compiler tự quyết, bạn không viết): Tensor Core yêu cầu mỗi thread
        # giữ đúng phần tử nào của ma trận (layout "MMA"), khác layout tối ưu cho đọc HBM.
        
        # Nên compiler chèn bước đổi layout, thường đi qua [SHARED MEM]:
        #     thanh ghi --(st.shared)--> shared mem --(ld.shared)--> thanh ghi (layout mới) --> mma
        # tl.trans(k) chỉ đổi cách "nhìn" ma trận (transpose), compiler xử lý trong layout.
        # -------------------------------------------------------------
        # Tensor Cores Matmul #1: QK^T inside SRAM
        # tl.dot uses NVIDIA Tensor Cores (MMA) - Matmul on Tensor Cores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)   # [32,32] fp32, trong thanh ghi
        qk += tl.dot(q, tl.trans(k)) * scale                   # dot: Tensor Core; "* scale": CUDA Core

        # Mask padding positions so they receive 0 probability
        # [CUDA CORE] Cột nào vượt N_CTX (padding) đặt -inf => exp(-inf)=0 => xác suất bằng 0
        col_ok = curr_n < N_CTX
        qk = tl.where(col_ok[None, :], qk, float("-inf"))

        # -------------------------------------------------------------
        # ONLINE SOFTMAX (toàn bộ chạy trên [CUDA CORE], dữ liệu trong [THANH GHI], fp32)
        # Vấn đề: softmax cần max/sum của CẢ HÀNG (128 cột), nhưng mỗi vòng chỉ thấy 32 cột.
        # Giải pháp: vừa đi vừa cập nhật max và tổng, rồi "quy đổi" kết quả cũ khi max đổi.
        # -------------------------------------------------------------
        # Online softmax statistics update (in fp32)
        m_ij = tl.max(qk, axis=1)             # [32]: max của từng hàng TRONG tile hiện tại (reduce giữa các thread)
        m_next = tl.maximum(m_i, m_ij)        # [32]: max mới = max(max cũ, max tile này)

        # Rescale factor for prior statistics
        alpha = tl.math.exp(m_i - m_next)     # rescale old results when max changes
                                              # [32]: hệ số quy đổi kết quả cũ sang mốc max mới (<=1)

        p = tl.math.exp(qk - m_next[:, None])  # [32,32]: "xác suất chưa chuẩn hóa" của tile này (trừ max để ổn định số học)

        # Rescale running accumulator
        acc = acc * alpha[:, None]            # quy đổi tích lũy cũ về mốc max mới

        # -------------------------------------------------------------
        # Matmul #2: acc += P @ V  -> chạy trên [TENSOR CORE]
        # p.to(float16): ép p từ fp32 xuống fp16 vì Tensor Core nhận đầu vào fp16;
        # kết quả tích lũy vẫn là fp32 (acc). Compiler lại có thể chèn [SHARED MEM]
        # để đổi layout của p và v trước khi mma, tương tự matmul #1.
        # -------------------------------------------------------------
        # Tensor Cores Matmul #2: P @ V inside SRAM
        acc += tl.dot(p.to(tl.float16), v)

        # Update running denominator and max  ([CUDA CORE], thanh ghi)
        l_i = l_i * alpha + tl.sum(p, axis=1)  # tổng exp mới = tổng cũ đã quy đổi + tổng của tile này
        m_i = m_next                           # lưu max mới cho vòng sau
        # Cuối vòng: qk, p, k, v của tile này bị bỏ/ghi đè. Ma trận S/P đầy đủ 128x128
        # KHÔNG HỀ tồn tại ở đâu, đây chính là điểm "fuse" tiết kiệm HBM.

    # -----------------------------------------------------------------
    # Step 3: Chuẩn hóa 1 lần ở cuối, rồi THANH GHI -> HBM
    # -----------------------------------------------------------------
    acc = acc / l_i[:, None]                    # normalize once at the end  ([CUDA CORE]) chia cho tổng exp của cả hàng

    # Lưới địa chỉ đầu ra O trong HBM (cùng cách dựng như q_ptrs)
    out_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # on-chip → HBM
    # DỮ LIỆU DI CHUYỂN: [THANH GHI] --(st.global)--> [HBM]. Đây là lần DUY NHẤT kernel ghi ra HBM.
    # .to(float16): ép acc fp32 -> fp16 cho khớp dtype của tensor O.
    tl.store(out_ptrs, acc.to(tl.float16), mask=q_mask)


# =====================================================================
# 2. PYTHON WRAPPER  (chạy trên CPU, chuẩn bị rồi "phóng" kernel lên GPU)
# =====================================================================
# WRAPPER này chạy trên CPU (Python thường), KHÔNG phải kernel. Việc của nó chỉ là:
#   (1) đọc shape, (2) tính scale, (3) cấp phát tensor đầu ra O trong HBM,
#   (4) tính grid, (5) "phóng" 1 fused kernel duy nhất lên GPU. Không có phép tính attention nào ở đây.
# Lệnh phóng kernel là BẤT ĐỒNG BỘ: CPU gửi lệnh rồi chạy tiếp, GPU làm việc ở phía sau.
# Synchronize: chờ GPU thực thi xong (vì launch là asynchronous)

# Input: q, k, v là torch tensor fp16 ĐÃ nằm sẵn trong HBM của GPU - được tạo sẵn ở main () - (không có copy dữ liệu ở đây);
#   block_m, block_n: kích thước tile (truyền xuống kernel thành BLOCK_M, BLOCK_N).
def flash_attention_2(q, k, v, block_m=32, block_n=32):
    """
    q, k, v: [Batch, Heads, N_CTX, HEAD_DIM] in float16 on CUDA.
    """
    # Bóc shape [Batch, Heads, N_CTX, HEAD_DIM]. Z = batch size, H = số head (quy ước tên của Triton tutorial).
    Z, H, N_CTX, HEAD_DIM = q.shape
    scale = 1.0 / (HEAD_DIM ** 0.5)  # scalar Python (float), tính trên CPU, truyền xuống kernel như tham số thường - scale attention

    # empty_like: allocate O cùng shape/dtype/device với q, nhưng KHÔNG khởi tạo giá trị
    # => kernel sẽ ghi đè toàn bộ O bằng tl.store.
    o = torch.empty_like(q)          # cấp phát sẵn tensor đầu ra O trong HBM (chưa có dữ liệu)

    # GRID = số program (≈ số block) sẽ tính ở GPU, mỗi program được xếp vào 1 SM.
    # Trục 0: số khối Q theo chiều sequence | trục 1: số head | trục 2: số batch
    # => program_id(0), (1), (2) trong kernel lần lượt lấy giá trị theo 3 trục này.
    # Ý nghĩa 3 phần tử của grid (khớp với tl.program_id trong kernel):
    #   triton.cdiv(N_CTX, block_m) = ceil(128/32) = 4  -> program_id(0) = pid_m: Q tile thứ mấy theo chiều token
    #   H = 2 (số head)                                 -> program_id(1) = pid_h: head thứ mấy
    #   Z = 1 (batch size)                              -> program_id(2) = pid_z: batch thứ mấy
    # => Mỗi (Q tile, head, batch) là 1 program độc lập. Tổng = 4*2*1 = 8 program.
    # cdiv = chia làm tròn lên (nếu N_CTX = 100 thì cdiv(100,32)=4, tile cuối bị mask phần dư).
    
    # Dùng launch kernel - lệnh cho GPU bắt đầu chạy hàm kernel.
    # CPU sẽ định nghĩa grid -> truyền cho xưởng GPU tờ hướng dẫn (hàm kernel) rồi nhấn chạy
    grid = (triton.cdiv(N_CTX, block_m), H, Z)  # (4, 2, 1) = 8 programs
    


    # kernel[grid](...) = Launch kernel. Python truyền sang GPU:
    #   - q, k, v, o: Triton tự lấy CON TRỎ tới dữ liệu trong HBM (không copy dữ liệu)
    #   - *q.stride() ...: các stride (số nguyên) để kernel tính địa chỉ
    #   - Z, H, N_CTX, scale: tham số vô hướng
    #   - BLOCK_M, BLOCK_N, HEAD_DIM: hằng số compile-time (constexpr)
    # Lần gọi đầu Triton biên dịch (JIT) kernel; các lần sau dùng bản đã cache.
    

    ''' Dòng này nói với GPU: flash_attn_2_fwd_kernel[grid] - nghĩa là “chạy 8 bản sao grid - program của hàm kernel trên GPU đó

        kernel: tờ hướng dẫn nào
        [grid]: gọi bao nhiêu công nhân (program) ra làm, ở đây là 8 (4 , 2, 1)
        (q, k, v, ...): nguyên liệu để công nhân dùng

    Sau khi nhấn nút, CPU không chờ và chạy tiếp dòng code sau, còn GPU tự làm ở phía sau. Đó là lý do cuối script phải có torch.cuda.synchronize() để chờ GPU làm xong.'''
    
    kernel = flash_attn_2_fwd_kernel[grid](
        # (a) 4 tensor -> Triton tự đổi thành 4 CON TRỎ HBM, khớp Q_ptr, K_ptr, V_ptr, O_ptr (đúng thứ tự).
        q, k, v, o,
        # (b) q.stride() trả về tuple 4 số, vd (8192, 4096, 32, 1). Dấu * "bung" tuple ra 4 tham số liên tiếp,
        #     khớp stride_qz, stride_qh, stride_qm, stride_qk. Tương tự cho k, v, o.
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        # (c) 3 số nguyên scalar -> Z, H, N_CTX trong kernel (Z = batch - H = head attention)
        Z, H, N_CTX,
        # (d) 1 số thực scalar -> scale trong kernel
        scale,
        # (e) Tham số keyword cho constexpr: giá trị được "nướng cứng" vào mã máy lúc compile.
        #     Đổi block_m/block_n/HEAD_DIM => Triton compile lại một phiên bản kernel mới.
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=HEAD_DIM,
    )
    return o, kernel


# =====================================================================
# 3. VERIFICATION & HARDWARE INSPECTION  (kiểm chứng đúng sai + soi phần cứng)
# =====================================================================
def main():
    torch.cuda.reset_peak_memory_stats()
    device_name = torch.cuda.get_device_name(0)

    # ---------------- Test configuration ----------------
    BATCH = 1             # = Z trong kernel: số batch độc lập xử lý cùng lúc.
    HEADS = 2             # = H trong kernel: số head attention. Mỗi head là 1 bài attention riêng (Q,K,V riêng),
                          #   không trao đổi dữ liệu giữa các head => 2 head = 2 nhóm program chạy song song.

    # N_CTX = độ dài sequence = số hàng của Q, K, V trong 1 (batch, head). Cắt thành tile 32 => 128/32 = 4 tile.
    # Q cắt 4 tile (mỗi tile giao 1 program), K và V cắt 4 tile (mỗi program duyệt cả 4 trong vòng for).
    N_CTX = 128           # 128 tokens = 4 tiles of 32 - NEVER IN HBM --> calculate in SRAM.
                          # (Chú ý: Q,K,V vẫn nằm trong HBM. Cái KHÔNG BAO GIỜ vào HBM là ma trận điểm S/P kích thước N×N.)

    # HEAD_DIM = số feature mỗi token trong 1 head (cột của Q,K,V). KHÔNG bị cắt thành tile:
    # Mỗi tile lấy đủ cả 32. (tl.dot yêu cầu chiều tích >= 16 để dùng Tensor Core.)
    HEAD_DIM = 32   

    # BLOCK_M = 32: mỗi tile Q gồm 32 TOKEN (hàng) liên tiếp trong 128 token, đủ 32 feature => tile [32×32].
    #   1 program phụ trách 1 tile Q => 128/32 = 4 program theo chiều token (mỗi head).
    BLOCK_M = 32          # Tile size for Q

    # BLOCK_N = 32: mỗi tile K/V gồm 32 TOKEN liên tiếp trong 128 token của K, V => tile [32×32].
    #   Mỗi vòng lặp for xử lý 1 tile (4 vòng cho 128 token). Tile càng lớn: ít vòng lặp, dùng nhiều thanh ghi/shared mem hơn.
    #   Tile nhỏ: ít tài nguyên nhưng nhiều vòng lặp hơn.
    BLOCK_N = 32          # Tile size for K/V

    print("=" * 80)
    print(f"FLASH ATTENTION 2 (32x32 TILES WITH TENSOR CORES & SHARED MEMORY SRAM)")
    print(f"Device: {device_name}")
    print(f"Shape: Batch={BATCH}, Heads={HEADS}, SeqLen={N_CTX}, Dim={HEAD_DIM}")
    print(f"Tile Sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"Grid: ({N_CTX // BLOCK_M} query blocks, {HEADS} heads, {BATCH} batch) = "
          f"{N_CTX // BLOCK_M * HEADS * BATCH} programs")
    print("=" * 80)

    # ---------------- A. Create test tensors ----------------
    # Tạo Q, K, V ngẫu nhiên fp16 trực tiếp trong HBM của GPU (device=cuda)
    torch.manual_seed(42)
    # Shape mỗi tensor = [BATCH, HEADS, N_CTX, HEAD_DIM] = [1, 2, 128, 32] => 8192 phần tử fp16 (16 KB) mỗi tensor.
    # Cùng 1 shape cho Q, K, V (self-attention). Dữ liệu sinh ngẫu nhiên,
    # Ở HBM của GPU.

 
    # ---------------- B. Run Triton FlashAttention-2 ----------------
    # Truyền q, k, v (đang ở HBM) vào wrapper => wrapper phóng kernel => nhận về:
    #   o_triton: tensor kết quả [1,2,128,32] fp16 (nằm trong HBM, do kernel ghi)
    #   compiled: đối tượng kernel đã compile, dùng để soi n_regs, shared mem, mã PTX ở phần sau
    o_triton, compiled = flash_attention_2(q, k, v, block_m=BLOCK_M, block_n=BLOCK_N)
    torch.cuda.synchronize()   # chờ GPU chạy xong (kernel launch là bất đồng bộ)

    # ---------------- C. Standard PyTorch Attention - Matmul Pytorch cách tính thông thường (Để so sánh với Flash Attention Triton) ----------------
    # Cách làm "thường": mỗi dòng là 1 kernel riêng, ma trận S (N×N) và P (N×N)
    # được ghi/đọc qua HBM. Đây là baseline để so sánh độ chính xác (và là thứ FlashAttention tránh).
    scale = 1.0 / (HEAD_DIM ** 0.5)                                        # ** 0.5 là lũy thừa = căn
    s_ref = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale   # S = QK^T*scale  (ghi N×N ra HBM)
    p_ref = torch.softmax(s_ref, dim=-1)                                   # P = softmax(S)  (đọc/ghi N×N)
    o_ref = torch.matmul(p_ref, v.float()).to(torch.float16)               # O = P@V

    # ---------------- Check Correctness ----------------
    max_diff = (o_triton - o_ref).abs().max().item()
    mean_diff = (o_triton - o_ref).abs().mean().item()
    is_close = torch.allclose(o_triton, o_ref, atol=2e-3, rtol=2e-3)

    print("\n--- 1. CORRECTNESS CHECK ---")
    print(f"Max absolute error  : {max_diff:.4e}")
    print(f"Mean absolute error : {mean_diff:.4e}")

    print(f"Flash Attention Result : {o_triton:.9e}")
    print(f"Attention Result : {o_ref:.9e}")
    print(f"Result              : {'PASS (Identical to Standard Attention)' if is_close else 'FAIL'}")

    # ---------------- Memory Allocation ----------------
    vram_bytes = torch.cuda.max_memory_allocated()
    print("\n--- 2. VRAM USAGE ---")
    print(f"Peak VRAM allocated : {vram_bytes / 1024:.2f} KB  ({vram_bytes / 1024**2:.4f} MB)")
    print(f"Total GPU VRAM limit: ~{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("-> Test memory footprint is minuscule and perfectly safe for 16GB VRAM.")

    # ---------------- Hardware & Compiler Inspection ----------------
    # Đọc thông số phần cứng mà compiler đã chọn cho kernel:
    #   n_regs      : số thanh ghi mỗi thread dùng
    #   n_spills    : số thanh ghi bị "tràn" ra bộ nhớ chậm (càng 0 càng tốt)
    #   shared      : số byte shared memory (SRAM) mỗi program dùng - do compiler tự chèn quanh tl.dot
    #   num_warps   : số warp mỗi program (mỗi warp = 32 thread) - Triton tự chia tile cho các thread này
    print("\n--- 3. COMPILED KERNEL HARDWARE STATS ---")
    md = getattr(compiled, "metadata", None)
    shared_mem = getattr(md, "shared", None)
    n_regs = getattr(compiled, "n_regs", "n/a")
    n_spills = getattr(compiled, "n_spills", "n/a")
    num_warps = getattr(md, "num_warps", "n/a")

    print(f"Registers per thread : {n_regs}")
    print(f"Register spills      : {n_spills}")
    print(f"Shared Memory (SRAM) : {shared_mem} bytes ({shared_mem / 1024:.2f} KB) per program block")
    print(f"Warps per block      : {num_warps} ({num_warps * 32} threads per block)")

    # ---------------- Save and Inspect PTX / IR ----------------
    # Lưu các tầng mã trung gian của compiler để bạn tự mở xem:
    #   ttir  : Triton IR (mức cao, còn tt.load / tt.dot / tt.store)
    #   ttgir : Triton GPU IR (đã gán layout cho thread/warp; tìm local_alloc / convert_layout = chỗ dùng shared memory)
    #   ptx   : mã assembly của NVIDIA (ld.global, st.shared, ld.shared, mma.sync...)
    saved_files = []
    for stage in ("ttir", "ttgir", "ptx"):
        if stage in compiled.asm:
            fname = f"kernel_tile32_{stage}.txt"
            with open(fname, "w") as f:
                f.write(compiled.asm[stage])
            saved_files.append(fname)

    print("\n--- 4. PTX ASSEMBLY INSTRUCTION ANALYSIS ---")
    if "ptx" in compiled.asm:
        ptx_code = compiled.asm["ptx"]
        lines = ptx_code.splitlines()

        # Đếm các lệnh quan trọng trong PTX:
        #   ld.global / st.global : đọc/ghi bộ nhớ global (HBM, có thể trúng L2/L1 cache)
        #   ld.shared / st.shared : đọc/ghi Shared Memory (SRAM trên chip)
        #   mma / wgmma           : lệnh nhân ma trận của Tensor Core
        ld_global_count = sum(1 for line in lines if "ld.global" in line)
        st_global_count = sum(1 for line in lines if "st.global" in line)
        ld_shared_count = sum(1 for line in lines if "ld.shared" in line)
        st_shared_count = sum(1 for line in lines if "st.shared" in line)
        mma_count = sum(1 for line in lines if ("mma." in line or "wgmma." in line))

        print(f"Total PTX lines      : {len(lines)}")
        print(f"Shared Memory Loads  (ld.shared) : {ld_shared_count}  (SRAM read)")
        print(f"Shared Memory Stores (st.shared) : {st_shared_count}  (SRAM write)")
        print(f"Tensor Core Matmuls  (mma/wgmma) : {mma_count}  (Hardware Tensor Cores activated!)")
        print(f"Global HBM Loads     (ld.global) : {ld_global_count}  (Q, K, V reads from HBM)")
        print(f"Global HBM Stores    (st.global) : {st_global_count}  (Only final O written to HBM!)")

        print("\nVerification of FlashAttention Principles:")
        if shared_mem and shared_mem > 0:
            print("  [YES] Shared Memory SRAM is actively allocated (> 0 bytes).")
        if mma_count > 0:
            print("  [YES] Hardware Tensor Cores (MMA) are actively used via tl.dot.")
        if st_shared_count > 0 or ld_shared_count > 0:
            print("  [YES] Data transfers through on-chip Shared Memory SRAM.")
        print(f"  [YES] Saved inspection files: {', '.join(saved_files)}")

    print("\n" + "=" * 80)
    print("Done! You can run this directly with: python3 flash_att_2_tile32x32.py")
    print("=" * 80)


if __name__ == "__main__":
    main()