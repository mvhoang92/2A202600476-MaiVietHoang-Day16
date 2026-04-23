# Kế hoạch Triển khai: Lab 16 — Reflexion Agent

## Tổng quan

Hoàn thiện hệ thống Reflexion Agent từ scaffold có sẵn: thay thế mock runtime bằng OpenAI GPT-4o-mini thật, triển khai vòng lặp Reflexion, chạy benchmark ≥100 mẫu HotpotQA, và xuất báo cáo đúng định dạng để `autograde.py` chấm điểm.

## Tasks

- [ ] 1. Hoàn thiện schemas trong `schemas.py`
  - [ ] 1.1 Định nghĩa `JudgeResult` với đầy đủ các trường
    - Thêm trường `score: Literal[0, 1]` — dùng `Literal` để Pydantic tự reject giá trị ngoài `{0, 1}`
    - Thêm trường `reason: str`
    - Thêm trường `missing_evidence: list[str] = []`
    - Thêm trường `spurious_claims: list[str] = []`
    - _Yêu cầu: 1.1, 1.2, 1.3, 1.4, 1.6_

  - [ ]* 1.2 Viết property test cho `JudgeResult` — Property 1: Score validation
    - **Property 1: Score validation**
    - **Validates: Yêu cầu 1.1, 1.6**
    - Dùng `hypothesis` với `st.integers().filter(lambda x: x not in {0, 1})` để kiểm tra mọi giá trị ngoài `{0, 1}` đều raise `ValidationError`
    - File: `tests/test_schemas.py`

  - [ ] 1.3 Định nghĩa `ReflectionEntry` với đầy đủ các trường
    - Thêm trường `attempt_id: int`
    - Thêm trường `failure_reason: str`
    - Thêm trường `lesson: str`
    - Thêm trường `next_strategy: str`
    - _Yêu cầu: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 1.4 Viết property test cho `ReflectionEntry` — Property 2: ReflectionEntry completeness
    - **Property 2: ReflectionEntry completeness**
    - **Validates: Yêu cầu 2.1, 2.5**
    - Dùng `st.builds(ReflectionEntry, ...)` để kiểm tra tất cả bốn trường đều non-empty
    - File: `tests/test_schemas.py`

  - [ ]* 1.5 Viết unit tests cho schemas
    - Test `JudgeResult` với `score=1` và `score=0`
    - Test giá trị mặc định của `missing_evidence` và `spurious_claims`
    - Test `ReflectionEntry` với đầy đủ bốn trường
    - File: `tests/test_schemas.py`
    - _Yêu cầu: 1.3, 1.4, 1.5, 2.2, 2.3, 2.4_

- [ ] 2. Viết System Prompts trong `prompts.py`
  - [ ] 2.1 Viết `ACTOR_SYSTEM` prompt
    - Hướng dẫn Actor đọc toàn bộ context trước khi trả lời
    - Yêu cầu trả lời ngắn gọn, trực tiếp (không giải thích dài dòng)
    - Hướng dẫn tham khảo `reflection_memory` khi không rỗng để tránh lặp lỗi cũ
    - Viết bằng tiếng Anh
    - _Yêu cầu: 3.1, 3.2, 3.3, 3.4_

  - [ ] 2.2 Viết `EVALUATOR_SYSTEM` prompt
    - Yêu cầu so sánh sau khi normalize (bỏ dấu câu, viết thường)
    - Yêu cầu trả về JSON hợp lệ với trường `score` (0 hoặc 1) và `reason`
    - Quy định `score=1` khi đúng, `score=0` khi sai
    - Hướng dẫn fallback `score=0` kèm lý do khi không thể phân tích
    - _Yêu cầu: 4.1, 4.2, 4.3, 4.4_

  - [ ] 2.3 Viết `REFLECTOR_SYSTEM` prompt
    - Yêu cầu phân tích lý do thất bại từ `JudgeResult.reason`
    - Yêu cầu trả về JSON hợp lệ với đủ bốn trường của `ReflectionEntry`
    - Hướng dẫn đề xuất chiến thuật cụ thể, actionable (không chung chung)
    - Viết bằng tiếng Anh
    - _Yêu cầu: 5.1, 5.2, 5.3, 5.4_

- [ ] 3. Tạo `llm_runtime.py` — thay thế mock bằng OpenAI thật
  - [ ] 3.1 Khởi tạo module và OpenAI client
    - Tạo file `src/reflexion_lab/llm_runtime.py`
    - Load API key từ `.env` qua `python-dotenv`
    - Khởi tạo `openai.OpenAI()` client với model `gpt-4o-mini`
    - Thêm hàm `extract_json(text: str) -> dict` để xử lý JSON lồng trong markdown hoặc có text thừa
    - _Yêu cầu: 6.4_

  - [ ] 3.2 Implement hàm `actor_answer()`
    - Signature: `actor_answer(example, attempt_id, agent_type, reflection_memory) -> tuple[str, int, float]`
    - Build user message từ `example.question`, `example.context`, và `reflection_memory`
    - Gọi OpenAI API với `ACTOR_SYSTEM` prompt
    - Đo latency bằng `time.time()` trước/sau API call
    - Đọc `response.usage.total_tokens` cho token count
    - Trả về `(answer, token_count, latency_ms)`; fallback `("", 0, 0.0)` khi lỗi
    - _Yêu cầu: 6.1, 6.6, 10.1_

  - [ ] 3.3 Implement hàm `evaluator()`
    - Signature: `evaluator(example, answer) -> tuple[JudgeResult, int, float]`
    - Gọi OpenAI API với `EVALUATOR_SYSTEM` prompt và `response_format={"type": "json_object"}`
    - Parse JSON response thành `JudgeResult` qua Pydantic validation
    - Dùng `extract_json()` khi parse thất bại; fallback `JudgeResult(score=0, reason="evaluation error")` khi vẫn lỗi
    - Retry 1 lần với exponential backoff khi API lỗi
    - _Yêu cầu: 6.2, 6.5, 6.6_

  - [ ] 3.4 Implement hàm `reflector()`
    - Signature: `reflector(example, attempt_id, judge) -> tuple[ReflectionEntry, int, float]`
    - Gọi OpenAI API với `REFLECTOR_SYSTEM` prompt và `response_format={"type": "json_object"}`
    - Parse JSON response thành `ReflectionEntry` qua Pydantic validation
    - Fallback `ReflectionEntry(attempt_id=N, failure_reason="reflection error", lesson="", next_strategy="")` khi lỗi
    - _Yêu cầu: 6.3, 6.5, 6.6_

  - [ ]* 1.6 Viết property test cho `evaluator()` — Property 3: Evaluator output validity
    - **Property 3: Evaluator output is always a valid JudgeResult**
    - **Validates: Yêu cầu 4.2, 6.2, 6.5**
    - Mock OpenAI client trả về malformed JSON; kiểm tra hàm vẫn trả về `JudgeResult` hợp lệ
    - File: `tests/test_llm_runtime.py`

  - [ ]* 1.7 Viết property test cho `reflector()` — Property 4: Reflector output validity
    - **Property 4: Reflector output is always a valid ReflectionEntry**
    - **Validates: Yêu cầu 5.2, 6.3, 6.5**
    - Mock OpenAI client trả về malformed JSON; kiểm tra hàm vẫn trả về `ReflectionEntry` hợp lệ
    - File: `tests/test_llm_runtime.py`

  - [ ]* 1.8 Viết property test cho token count — Property 9: Token count from API usage
    - **Property 9: Token count from API usage**
    - **Validates: Yêu cầu 6.6, 10.1**
    - Mock response với `usage.total_tokens = N`; kiểm tra `token_count` trả về đúng bằng `N`
    - File: `tests/test_llm_runtime.py`

- [ ] 4. Checkpoint — Kiểm tra schemas, prompts và llm_runtime
  - Đảm bảo tất cả tests trong `tests/test_schemas.py` và `tests/test_llm_runtime.py` pass
  - Hỏi người dùng nếu có vấn đề cần làm rõ trước khi tiếp tục

- [ ] 5. Triển khai Reflexion Loop trong `agents.py`
  - [ ] 5.1 Cập nhật import để hỗ trợ cả `llm_runtime` và `mock_runtime`
    - Thêm tham số `runtime` vào `BaseAgent` (mặc định là `"mock"`)
    - Import có điều kiện: `runtime="real"` dùng `llm_runtime`, `runtime="mock"` dùng `mock_runtime`
    - _Yêu cầu: 6.1, 6.2, 6.3_

  - [ ] 5.2 Implement logic Reflexion trong `BaseAgent.run()`
    - Thay thế token/latency hardcode bằng giá trị thực từ `llm_runtime` (tuple return)
    - Khi `agent_type == "reflexion"` và `judge.score == 0` và `attempt_id < max_attempts`: gọi `reflector()`, thêm `next_strategy` vào `reflection_memory`, thêm `ReflectionEntry` vào `reflections`
    - Cộng token/latency của `reflector()` vào `trace` cuối cùng
    - Dừng vòng lặp ngay khi `judge.score == 1`
    - _Yêu cầu: 7.1, 7.2, 7.3, 7.4, 7.6, 7.7_

  - [ ] 5.3 Đảm bảo `ReActAgent` chỉ chạy đúng 1 lần thử
    - Xác nhận `max_attempts=1` và không gọi `reflector()` trong bất kỳ trường hợp nào
    - _Yêu cầu: 7.5_

  - [ ]* 5.4 Viết property test cho memory propagation — Property 5
    - **Property 5: Reflexion memory propagation**
    - **Validates: Yêu cầu 7.1, 7.2**
    - Mock runtime; kiểm tra `next_strategy` từ mỗi `ReflectionEntry` xuất hiện trong `reflection_memory` của lần thử tiếp theo
    - File: `tests/test_agents.py`

  - [ ]* 5.5 Viết property test cho early stopping — Property 6
    - **Property 6: Early stopping on success**
    - **Validates: Yêu cầu 7.4**
    - Mock evaluator trả về `score=1` tại attempt `k`; kiểm tra `RunRecord` có đúng `k` traces và `reflector()` không được gọi sau attempt `k`
    - File: `tests/test_agents.py`

  - [ ]* 5.6 Viết property test cho ReAct single attempt — Property 7
    - **Property 7: ReAct agent single attempt invariant**
    - **Validates: Yêu cầu 7.5**
    - Dùng `st.builds(QAExample, ...)` với mock runtime; kiểm tra `RunRecord` luôn có đúng 1 `AttemptTrace`
    - File: `tests/test_agents.py`

  - [ ]* 5.7 Viết property test cho token aggregation — Property 8
    - **Property 8: Token estimate aggregation invariant**
    - **Validates: Yêu cầu 7.6, 10.2**
    - Kiểm tra `RunRecord.token_estimate == sum(t.token_estimate for t in traces)` với nhiều bộ token ngẫu nhiên
    - File: `tests/test_agents.py`

- [ ] 6. Tải dataset HotpotQA ≥100 mẫu
  - [ ] 6.1 Tạo script `scripts/download_hotpot.py`
    - Dùng `datasets` library (HuggingFace) để load `hotpot_qa` split `validation`
    - Convert 150 mẫu đầu sang format `QAExample` (giới hạn 4 context chunks mỗi mẫu)
    - Lưu vào `data/hotpot_100.json` với encoding UTF-8
    - _Yêu cầu: 8.3_

  - [ ]* 6.2 Viết integration test kiểm tra dataset
    - Kiểm tra `load_dataset("data/hotpot_100.json")` trả về ≥100 mẫu hợp lệ
    - Kiểm tra mỗi mẫu có đủ các trường `qid`, `question`, `gold_answer`, `context`
    - File: `tests/test_integration.py`
    - _Yêu cầu: 8.3_

- [ ] 7. Cập nhật `run_benchmark.py` — thêm `--mode` flag và error handling
  - [ ] 7.1 Thêm tham số `--mode` vào CLI
    - Thêm `mode: str = "mock"` vào hàm `main()`
    - Khi `mode="real"`: khởi tạo agents với `runtime="real"` và dataset `data/hotpot_100.json`
    - Khi `mode="mock"`: giữ nguyên behavior hiện tại với `mock_runtime`
    - Truyền `mode` vào `build_report()`
    - _Yêu cầu: 8.1, 8.2_

  - [ ] 7.2 Thêm error handling cho từng sample
    - Bọc `agent.run(example)` trong `try/except`
    - Ghi log lỗi với format `[i/total] qid=... error: ...` và `continue` sang sample tiếp theo
    - _Yêu cầu: 8.5_

  - [ ]* 7.3 Viết property test cho error resilience — Property 10
    - **Property 10: Error resilience in benchmark**
    - **Validates: Yêu cầu 8.5**
    - Inject lỗi vào một subset samples; kiểm tra benchmark vẫn hoàn thành và trả về records cho các samples không lỗi
    - File: `tests/test_benchmark.py`

- [ ] 8. Checkpoint — Kiểm tra toàn bộ pipeline với mock mode
  - Chạy `python run_benchmark.py --mode mock` và xác nhận `report.json` được tạo đúng
  - Đảm bảo tất cả tests pass
  - Hỏi người dùng nếu có vấn đề trước khi chạy benchmark thật

- [ ] 9. Viết tests bổ sung và kiểm tra report
  - [ ] 9.1 Viết property test cho report token sum — Property 11
    - **Property 11: Report token sum consistency**
    - **Validates: Yêu cầu 9.3**
    - Dùng `st.lists(st.builds(RunRecord, ...))` để kiểm tra `avg_token_estimate` trong summary bằng mean thực tế
    - File: `tests/test_reporting.py`

  - [ ]* 9.2 Viết integration test cho benchmark pipeline
    - Dùng mock mode với mini dataset; kiểm tra cả hai agent chạy và tạo file output
    - Kiểm tra `report.json` có đủ sáu trường bắt buộc
    - File: `tests/test_integration.py`
    - _Yêu cầu: 8.1, 8.4, 9.1_

  - [ ]* 9.3 Viết smoke tests
    - Kiểm tra `OPENAI_API_KEY` được load từ `.env`
    - Kiểm tra các prompts được viết bằng tiếng Anh (có chứa từ tiếng Anh cơ bản)
    - File: `tests/test_smoke.py`
    - _Yêu cầu: 3.4, 5.4, 6.4_

- [ ] 10. Chạy benchmark thực tế và tạo report
  - [ ] 10.1 Cài đặt dependencies bổ sung
    - Thêm `openai`, `hypothesis`, `datasets` vào `requirements.txt`
    - Xác nhận `OPENAI_API_KEY` có trong `.env`
    - _Yêu cầu: 6.4_

  - [ ] 10.2 Chạy benchmark với `--mode real`
    - Chạy `python run_benchmark.py --mode real --dataset data/hotpot_100.json --out-dir outputs/real_run`
    - Xác nhận `report.json` có `meta.num_records >= 100`
    - Xác nhận `report.json` có `meta.mode == "real"`
    - _Yêu cầu: 8.1, 8.2, 8.3_

  - [ ] 10.3 Kiểm tra report đạt tiêu chí autograde
    - Chạy `python autograde.py --report-path outputs/real_run/report.json`
    - Xác nhận schema completeness (6/6 trường bắt buộc)
    - Xác nhận `examples` có ≥20 bản ghi
    - Xác nhận `discussion` có ≥250 ký tự
    - Xác nhận `failure_modes` có ≥3 loại
    - _Yêu cầu: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 10.4 Cập nhật `discussion` trong `reporting.py`
    - Viết phân tích thực nghiệm dựa trên kết quả benchmark thật (≥250 ký tự)
    - Giải thích khi nào reflection memory hữu ích, failure modes nào còn tồn tại
    - _Yêu cầu: 9.6_

- [ ] 11. Checkpoint cuối — Đảm bảo tất cả tests pass và report hợp lệ
  - Chạy toàn bộ test suite và xác nhận không có lỗi
  - Chạy `autograde.py` lần cuối để xác nhận điểm
  - Hỏi người dùng nếu có vấn đề cần xử lý

- [ ] 12. (Bonus) Triển khai `structured_evaluator`
  - [ ] 12.1 Thêm hàm `structured_evaluator()` vào `llm_runtime.py`
    - Dùng `client.beta.chat.completions.parse()` với `response_format=JudgeResult` (Pydantic model trực tiếp)
    - Không cần bước parse JSON thủ công
    - _Yêu cầu: 11.1, 11.3_

  - [ ] 12.2 Thêm `"structured_evaluator"` vào `ReportPayload.extensions`
    - Cập nhật `reporting.py` để include extension khi `structured_evaluator` được dùng
    - _Yêu cầu: 11.2_

- [ ] 13. (Bonus) Triển khai `GlobalReflectionMemory`
  - [ ] 13.1 Tạo class `GlobalReflectionMemory` trong `agents.py`
    - Lưu `lesson` từ các câu hỏi đã xử lý trước đó (max 10 lessons)
    - Hàm `get_relevant(question, k=3)` trả về `k` lessons gần nhất
    - _Yêu cầu: 12.1, 12.2_

  - [ ] 13.2 Tích hợp `GlobalReflectionMemory` vào `ReflexionAgent`
    - Truyền lessons liên quan vào `reflection_memory` của Actor khi xử lý câu hỏi mới
    - Thêm `"reflection_memory"` vào `ReportPayload.extensions`
    - _Yêu cầu: 12.2, 12.3_

## Ghi chú

- Tasks đánh dấu `*` là tùy chọn, có thể bỏ qua để triển khai nhanh hơn
- Mỗi task tham chiếu đến yêu cầu cụ thể để đảm bảo traceability
- Các checkpoint giúp kiểm tra tiến độ theo từng giai đoạn
- Property tests dùng thư viện `hypothesis` với `max_examples=100`
- Chạy tests bằng: `pytest tests/ -v` (không dùng watch mode)
