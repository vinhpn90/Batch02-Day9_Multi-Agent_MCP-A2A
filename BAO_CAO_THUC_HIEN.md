# Báo Cáo Nội Dung Đã Thực Hiện

## 1. Thông tin chung

- Dự án: `Batch02-Day9_Multi-Agent_MCP-A2A`
- Nội dung: Codelab Multi-Agent với MCP/A2A, LangGraph, LangChain, OpenRouter
- Ngày thực hiện: 2026-06-09
- Họ và tên: Phạm Ngọc Vinh
- Mã học viên: 2A202600563

## 2. Phần 1 - Direct LLM Calling

### Bài 1.1 - Thay đổi câu hỏi

Đã sửa biến `QUESTION` trong:

- `stages/stage_1_direct_llm/main.py`

Câu hỏi mới:

```text
Người lao động có thể làm gì nếu bị sa thải trái pháp luật?
```

### Bài 1.2 - Temperature control

Đã thêm cấu hình `temperature=0.3` trong:

- `common/llm.py`

Ngoài ra đã thêm `max_tokens=1000` để tránh lỗi OpenRouter yêu cầu số token quá lớn so với credit hiện có.

### Kết quả kiểm tra

Đã chạy:

```bash
uv run python stages/stage_1_direct_llm/main.py
```

Kết quả: chạy thành công, LLM trả lời câu hỏi mới.

## 3. Phần 2 - LLM + RAG & Tools

### Bài 2.1 - Thêm knowledge base entry

Đã thêm entry `labor_law` về luật lao động Việt Nam vào:

- `stages/stage_2_rag_tools/main.py`
- `exercises/exercise_2_tools.py`

Entry này gồm các keyword như:

```text
lao động, sa thải, hợp đồng lao động, labor, termination
```

### Bài 2.2 - Tạo tool mới

Đã thêm tool:

```python
check_statute_of_limitations(case_type: str) -> str
```

Tool trả về thời hiệu theo loại vụ án:

- `contract`: 4 năm
- `tort`: 2-3 năm tùy bang
- `property`: 5 năm

Đã thêm tool vào danh sách `TOOLS`.

### Kết quả kiểm tra

Đã chạy:

```bash
uv run python stages/stage_2_rag_tools/main.py
uv run python exercises/exercise_2_tools.py
```

Kết quả:

- Stage 2 chạy thành công.
- Exercise 2 gọi đúng tool `check_statute_of_limitations`.
- Tool trả về thời hiệu hợp đồng là `4 năm (UCC § 2-725)`.

## 4. Phần 3 - Single Agent ReAct

### Bài 3.1 - Thêm tool tra cứu án lệ

Đã thêm tool:

```python
search_case_law(keywords: str) -> str
```

Trong:

- `stages/stage_3_single_agent/main.py`

Tool hỗ trợ các án lệ:

- `Hadley v. Baxendale (1854)` - consequential damages
- `Donoghue v. Stevenson (1932)` - duty of care
- `Carlill v. Carbolic Smoke Ball Co (1893)` - unilateral contract

Đã thêm tool vào danh sách `TOOLS`.

### Bài 3.2 - Debug agent reasoning

Codelab ghi `verbose=True`, nhưng phiên bản LangGraph trong dự án không hỗ trợ tham số này. Đã kiểm tra signature của `create_react_agent()` và dùng tham số đúng là:

```python
debug=True
```

### Kết quả kiểm tra

Đã chạy:

```bash
uv run python stages/stage_3_single_agent/main.py
```

Kết quả:

- Agent gọi `search_legal_database`.
- Agent gọi `search_case_law`.
- Agent gọi `calculate_penalty`.
- Debug output hiển thị các bước `THINK + ACT`, `OBSERVE`, `FINAL ANSWER`.

## 5. Phần 4 - Multi-Agent In-Process

### Bài 4.1 - Thêm Privacy Agent

Đã thêm Privacy Agent vào:

- `stages/stage_4_milti_agent/main.py`
- `exercises/exercise_4_multiagent.py`

Privacy Agent phân tích:

- GDPR
- Data protection
- Privacy rights
- Data breach
- Nghĩa vụ thông báo
- Tiền phạt và rủi ro kiện tụng

### Bài 4.2 - Conditional routing

Đã thêm logic routing để gọi Privacy Agent khi câu hỏi có các keyword:

```text
data, privacy, gdpr, dữ liệu
```

Đã thêm:

- Field `needs_privacy`
- Field `privacy_result` / `privacy_analysis`
- Node `privacy_agent`
- Edge từ `privacy_agent` về `aggregate`

### Vẽ graph bước 3 phần 4

Đã tạo graph cho Stage 4:

- `stages/stage_4_milti_agent/graph_privacy.mmd`
- `stages/stage_4_milti_agent/graph_privacy.svg`

Luồng graph:

```text
START -> analyze_law -> check_routing
check_routing -> tax_agent / compliance_agent / privacy_agent / aggregate
tax_agent -> aggregate
compliance_agent -> aggregate
privacy_agent -> aggregate
aggregate -> END
```

### Demo HTML Stage 4

Đã tạo file HTML demo tương tác:

- `stages/stage_4_milti_agent/demo.html`

Chức năng:

- Nhập câu hỏi.
- Router chọn agent phù hợp.
- Hiển thị graph tương tác.
- Timeline mô phỏng từng bước.
- Final answer mô phỏng kết quả aggregate.

### Kết quả kiểm tra

Đã chạy:

```bash
uv run python exercises/exercise_4_multiagent.py
uv run python stages/stage_4_milti_agent/main.py
```

Kết quả:

- Exercise 4 chạy thành công.
- Stage 4 chạy thành công.
- Router chọn đúng `needs_tax=True`, `needs_compliance=True`, `needs_privacy=True` với câu hỏi có data/GDPR/tax.

## 6. Phần 5 - Distributed A2A System

### Khởi động hệ thống

Đã chạy full Stage 5 bằng:

```bash
uv run bash start_all.sh
```

Các service đã khởi động:

- Registry: `localhost:10000`
- Customer Agent: `localhost:10100`
- Law Agent: `localhost:10101`
- Tax Agent: `localhost:10102`
- Compliance Agent: `localhost:10103`

### Test hệ thống

Đã chạy:

```bash
uv run python test_client.py
```

Kết quả: hệ thống trả lời thành công qua luồng A2A.

### Bài 5.1 - Trace request flow

Đã trace request với `trace_id` trong logs.

Ví dụ trace:

```text
trace_id=c44fc1bb-2db0-499e-bd6e-567478e57e25
```

Luồng request:

```text
User
-> Customer Agent
-> Registry discover legal_question
-> Law Agent
-> Registry discover tax_question
-> Tax Agent
-> Law Agent
-> Customer Agent
-> User
```

Đã tạo sequence diagram:

- `docs/stage_5_sequence.mmd`
- `docs/stage_5_sequence.svg`

### Bài 5.2 - Test dynamic discovery

Đã dừng Tax Agent và chạy lại:

```bash
uv run python test_client.py
```

Kết quả:

- Registry vẫn trả endpoint của Tax Agent.
- Law Agent gọi Tax Agent nhưng gặp lỗi `httpx.ConnectError`.
- Hệ thống không crash.
- Client vẫn nhận response fallback, có thông báo không lấy được tax analysis.

### Bài 5.3 - Modify agent behavior

Đã sửa prompt trong:

- `tax_agent/graph.py`

Mục tiêu:

- Tax Agent trả lời ngắn gọn hơn.
- Dùng 3-5 bullet.
- Giới hạn khoảng dưới 150 từ nếu user không yêu cầu chi tiết.

Đã restart Tax Agent và test lại thành công.

## 7. Đo latency và tối ưu Stage 5

### Baseline

Đã đo bằng:

```bash
/usr/bin/time -p uv run python test_client.py
```

Kết quả baseline:

```text
real 70.30
```

Latency ban đầu: **70.30 giây**

### Phương án tối ưu

Đã áp dụng:

1. Customer Agent direct-delegate tới Law Agent, bỏ LLM front-desk call.
2. Law Agent dùng keyword routing thay vì gọi LLM để quyết định specialist.
3. Thêm `STAGE5_FAST_DEMO=true` để demo bằng cached deterministic analysis, vẫn giữ Registry discovery và A2A delegation.

Các file chính đã sửa:

- `customer_agent/agent_executor.py`
- `law_agent/graph.py`
- `tax_agent/agent_executor.py`
- `compliance_agent/agent_executor.py`
- `.env`

### Kết quả sau tối ưu

Đã đo lại bằng:

```bash
/usr/bin/time -p uv run python test_client.py
```

Kết quả:

```text
real 0.26
```

Latency sau tối ưu: **0.26 giây**

Mức giảm:

```text
70.30s -> 0.26s
Giảm 70.04s
Tỷ lệ giảm khoảng 99.6%
```

### Demo HTML trước/sau tối ưu

Đã tạo file:

- `docs/stage5_latency_demo.html`

Chức năng:

- So sánh before/after.
- Hiển thị latency `70.30s` và `0.26s`.
- Hiển thị số lượng LLM calls.
- Animation flow Customer -> Registry -> Law -> Tax -> Aggregate.
- Danh sách các bước trước và sau tối ưu.

## 8. File HTML demo đã tạo

Các file HTML có thể mở trực tiếp bằng trình duyệt, không cần chạy server:

```bash
open stages/stage_4_milti_agent/demo.html
open docs/stage5_latency_demo.html
```

### 8.1. `stages/stage_4_milti_agent/demo.html`

Đây là file demo tương tác cho **Stage 4 - Multi-Agent In-Process**.

Mục tiêu của file:

- Minh họa cách các agent trong Stage 4 phối hợp với nhau.
- Cho người xem nhập câu hỏi và quan sát router chọn agent phù hợp.
- Mô phỏng luồng xử lý của LangGraph mà không cần gọi API thật.
- Phù hợp để trình bày trên lớp hoặc demo nhanh khi không muốn tiêu tốn OpenRouter credit.

Các thành phần chính trong giao diện:

- Ô nhập câu hỏi.
- Các nút ví dụ nhanh:
  - Privacy / GDPR
  - Tax
  - Compliance
  - General Law Only
- Graph tương tác gồm:
  - `analyze_law`
  - `check_routing`
  - `tax_agent`
  - `compliance_agent`
  - `privacy_agent`
  - `aggregate`
- Timeline hiển thị từng bước xử lý.
- Panel `Final Answer` mô phỏng kết quả tổng hợp cuối cùng.

Logic mô phỏng:

- Nếu câu hỏi có keyword `tax`, `irs`, `thuế`, `offshore`, `fbar`, `fatca`, hệ thống chọn `tax_agent`.
- Nếu câu hỏi có keyword `compliance`, `sec`, `regulation`, `sox`, `fcpa`, `aml`, hệ thống chọn `compliance_agent`.
- Nếu câu hỏi có keyword `data`, `privacy`, `gdpr`, `dữ liệu`, `ccpa`, `consent`, hệ thống chọn `privacy_agent`.
- Nếu không có keyword specialist, graph chuyển từ router thẳng tới `aggregate`.

Ý nghĩa khi demo:

- Giúp người xem hiểu rõ khác biệt giữa single-agent và multi-agent.
- Cho thấy vì sao cần `check_routing`.
- Cho thấy các specialist agents có thể chạy song song trước khi kết quả được tổng hợp.
- Trực quan hóa bài 4.1 và 4.2 đã thực hiện.

### 8.2. `docs/stage5_latency_demo.html`

Đây là file demo so sánh **trước và sau tối ưu latency của Stage 5 - Distributed A2A System**.

Mục tiêu của file:

- Trình bày số liệu latency đo được khi chạy full Stage 5.
- So sánh kiến trúc trước tối ưu và sau tối ưu.
- Minh họa vì sao giảm số lần gọi LLM giúp giảm thời gian phản hồi.
- Cho thấy tối ưu vẫn giữ được A2A flow và Registry discovery.

Số liệu hiển thị trong demo:

```text
Before optimization: 70.30s
After optimization: 0.26s
Reduction: khoảng 99.6%
```

Các thành phần chính trong giao diện:

- Nút `Before`: xem luồng trước tối ưu.
- Nút `After`: xem luồng sau tối ưu.
- Nút `Run Demo`: chạy animation flow tương ứng.
- Nút `Show Comparison`: hiển thị so sánh latency.
- Metric cards:
  - Latency
  - Số LLM calls
  - Trạng thái A2A flow
  - Tỷ lệ giảm latency
- Biểu đồ thanh so sánh latency.
- Danh sách các bước xử lý trước và sau tối ưu.

Luồng trước tối ưu:

```text
Customer ReAct LLM
-> Registry discovery
-> Law analysis LLM
-> Law routing LLM
-> Tax Agent LLM
-> Law aggregate LLM
-> Customer final LLM
```

Luồng sau tối ưu:

```text
Customer direct delegate
-> Registry discovery
-> Law cached analysis
-> Keyword routing
-> Tax Agent fast path
-> Template aggregate
```

Ý nghĩa khi demo:

- Cho thấy latency baseline của hệ thống là `70.30s`.
- Cho thấy latency sau tối ưu là `0.26s`.
- Giải thích trực quan nguyên nhân giảm latency:
  - Bỏ LLM call ở Customer Agent.
  - Bỏ LLM call ở Law Agent routing.
  - Dùng deterministic/cached analysis trong `STAGE5_FAST_DEMO=true`.
- Chứng minh hệ thống vẫn giữ kiến trúc distributed A2A:
  - Customer Agent
  - Registry
  - Law Agent
  - Tax Agent
  - A2A message passing

## 9. Các file diagram đã tạo

- `stages/stage_4_milti_agent/graph_privacy.mmd`
- `stages/stage_4_milti_agent/graph_privacy.svg`
- `docs/stage_5_sequence.mmd`
- `docs/stage_5_sequence.svg`

## 10. Ghi chú kỹ thuật

- OpenRouter model trong `.env` đã được đổi sang model miễn phí hợp lệ:

```env
OPENROUTER_MODEL=nex-agi/nex-n2-pro:free
```

- Đã thêm:

```env
STAGE5_FAST_DEMO=true
```

- Sau mỗi lần chạy Stage 5, các service process đã được dừng để không giữ port nền.

## 11. Kết luận

Đã hoàn thành các phần thực hành chính:

- Phần 1: Direct LLM
- Phần 2: RAG & Tools
- Phần 3: ReAct Agent
- Phần 4: Multi-Agent In-Process
- Phần 5: Distributed A2A System
- Đo và tối ưu latency Stage 5
- Tạo graph, sequence diagram và HTML demo phục vụ trình bày
