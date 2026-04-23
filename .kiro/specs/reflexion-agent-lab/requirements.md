# Tài liệu Yêu cầu — Lab 16: Reflexion Agent

## Giới thiệu

Bài lab này yêu cầu học viên hoàn thiện một hệ thống **Reflexion Agent** từ scaffold có sẵn. Scaffold hiện dùng mock data để giả lập phản hồi LLM. Học viên phải thay thế toàn bộ phần mock bằng LLM thật, hoàn thiện schema dữ liệu, viết system prompt, triển khai vòng lặp Reflexion, đo token thực tế, chạy benchmark trên ≥100 mẫu HotpotQA, và xuất báo cáo đúng định dạng để công cụ `autograde.py` chấm điểm được.

---

## Bảng thuật ngữ

- **Reflexion_Agent**: Hệ thống agent học viên xây dựng, bao gồm Actor, Evaluator và Reflector.
- **Actor**: Thành phần LLM nhận câu hỏi và context, sinh ra câu trả lời.
- **Evaluator**: Thành phần LLM chấm điểm câu trả lời (0 hoặc 1) và trả về lý do.
- **Reflector**: Thành phần LLM phân tích lỗi từ Evaluator và đề xuất chiến thuật cho lần thử tiếp theo.
- **ReAct_Agent**: Agent chỉ thực hiện một lần thử duy nhất, không có vòng lặp reflection.
- **Reflexion_Loop**: Vòng lặp tối đa `max_attempts` lần: Actor → Evaluator → Reflector → Actor (nếu chưa đúng).
- **Reflection_Memory**: Danh sách các bài học (`lesson`) và chiến thuật (`next_strategy`) tích lũy qua các lần thử.
- **JudgeResult**: Schema Pydantic lưu kết quả đánh giá của Evaluator (score, reason, ...).
- **ReflectionEntry**: Schema Pydantic lưu một mục reflection (attempt_id, failure_reason, lesson, next_strategy).
- **HotpotQA**: Bộ dữ liệu hỏi đáp đa bước (multi-hop) dùng để benchmark.
- **EM (Exact Match)**: Tỉ lệ câu trả lời khớp chính xác với đáp án vàng sau khi chuẩn hóa.
- **Token_Count**: Số token thực tế trả về từ API LLM (không phải ước tính).
- **RunRecord**: Schema lưu toàn bộ kết quả một lần chạy cho một câu hỏi.
- **ReportPayload**: Schema lưu toàn bộ báo cáo benchmark (meta, summary, failure_modes, examples, extensions, discussion).
- **LLM_Runtime**: Module thay thế `mock_runtime.py`, gọi LLM thật qua API.

---

## Yêu cầu

### Yêu cầu 1: Hoàn thiện Schema JudgeResult

**User Story:** Là học viên, tôi muốn định nghĩa đầy đủ schema `JudgeResult`, để Evaluator có thể trả về kết quả có cấu trúc mà các thành phần khác trong hệ thống có thể sử dụng.

#### Tiêu chí chấp nhận

1. THE `JudgeResult` SHALL có trường `score` kiểu `int` nhận giá trị `0` hoặc `1`.
2. THE `JudgeResult` SHALL có trường `reason` kiểu `str` mô tả lý do chấm điểm.
3. WHERE học viên muốn hỗ trợ phân tích lỗi chi tiết, THE `JudgeResult` SHALL có trường `missing_evidence` kiểu `list[str]` với giá trị mặc định là danh sách rỗng.
4. WHERE học viên muốn hỗ trợ phân tích lỗi chi tiết, THE `JudgeResult` SHALL có trường `spurious_claims` kiểu `list[str]` với giá trị mặc định là danh sách rỗng.
5. WHEN `JudgeResult` được khởi tạo với `score=1`, THE `JudgeResult` SHALL cho phép `reason` là chuỗi không rỗng.
6. IF `score` nhận giá trị ngoài tập `{0, 1}`, THEN THE `JudgeResult` SHALL báo lỗi validation.

---

### Yêu cầu 2: Hoàn thiện Schema ReflectionEntry

**User Story:** Là học viên, tôi muốn định nghĩa đầy đủ schema `ReflectionEntry`, để Reflector có thể lưu lại bài học và chiến thuật sau mỗi lần thử thất bại.

#### Tiêu chí chấp nhận

1. THE `ReflectionEntry` SHALL có trường `attempt_id` kiểu `int` xác định lần thử tương ứng.
2. THE `ReflectionEntry` SHALL có trường `failure_reason` kiểu `str` mô tả nguyên nhân thất bại.
3. THE `ReflectionEntry` SHALL có trường `lesson` kiểu `str` chứa bài học rút ra.
4. THE `ReflectionEntry` SHALL có trường `next_strategy` kiểu `str` chứa chiến thuật cho lần thử tiếp theo.
5. WHEN một `ReflectionEntry` được tạo ra, THE `ReflectionEntry` SHALL có tất cả bốn trường trên được điền đầy đủ (không được để trống).

---

### Yêu cầu 3: Viết System Prompt cho Actor

**User Story:** Là học viên, tôi muốn viết system prompt cho Actor, để LLM biết cách sử dụng context và reflection memory để sinh câu trả lời chính xác.

#### Tiêu chí chấp nhận

1. THE `ACTOR_SYSTEM` prompt SHALL hướng dẫn Actor đọc toàn bộ context được cung cấp trước khi trả lời.
2. THE `ACTOR_SYSTEM` prompt SHALL yêu cầu Actor trả về câu trả lời ngắn gọn, trực tiếp (không giải thích dài dòng).
3. WHEN `reflection_memory` không rỗng, THE `ACTOR_SYSTEM` prompt SHALL hướng dẫn Actor tham khảo các bài học trong `reflection_memory` để tránh lặp lại lỗi cũ.
4. THE `ACTOR_SYSTEM` prompt SHALL được viết bằng tiếng Anh để tương thích với các LLM phổ biến.

---

### Yêu cầu 4: Viết System Prompt cho Evaluator

**User Story:** Là học viên, tôi muốn viết system prompt cho Evaluator, để LLM chấm điểm câu trả lời một cách nhất quán và trả về JSON có cấu trúc.

#### Tiêu chí chấp nhận

1. THE `EVALUATOR_SYSTEM` prompt SHALL yêu cầu Evaluator so sánh câu trả lời dự đoán với đáp án vàng sau khi chuẩn hóa (bỏ dấu câu, viết thường).
2. THE `EVALUATOR_SYSTEM` prompt SHALL yêu cầu Evaluator trả về JSON hợp lệ khớp với schema `JudgeResult` (có trường `score` và `reason`).
3. THE `EVALUATOR_SYSTEM` prompt SHALL quy định `score=1` khi câu trả lời đúng và `score=0` khi sai.
4. IF Evaluator không thể phân tích câu trả lời, THEN THE `EVALUATOR_SYSTEM` prompt SHALL hướng dẫn Evaluator trả về `score=0` kèm lý do cụ thể.

---

### Yêu cầu 5: Viết System Prompt cho Reflector

**User Story:** Là học viên, tôi muốn viết system prompt cho Reflector, để LLM phân tích lỗi và đề xuất chiến thuật cải thiện cho lần thử tiếp theo.

#### Tiêu chí chấp nhận

1. THE `REFLECTOR_SYSTEM` prompt SHALL yêu cầu Reflector phân tích lý do thất bại từ `JudgeResult.reason`.
2. THE `REFLECTOR_SYSTEM` prompt SHALL yêu cầu Reflector trả về JSON hợp lệ khớp với schema `ReflectionEntry` (có đủ bốn trường).
3. THE `REFLECTOR_SYSTEM` prompt SHALL hướng dẫn Reflector đề xuất chiến thuật cụ thể, có thể hành động được (actionable), không chung chung.
4. THE `REFLECTOR_SYSTEM` prompt SHALL được viết bằng tiếng Anh để tương thích với các LLM phổ biến.

---

### Yêu cầu 6: Xây dựng LLM Runtime thay thế Mock

**User Story:** Là học viên, tôi muốn thay thế `mock_runtime.py` bằng module gọi LLM thật, để hệ thống hoạt động với dữ liệu thực tế thay vì dữ liệu giả lập.

#### Tiêu chí chấp nhận

1. THE `LLM_Runtime` SHALL cung cấp hàm `actor_answer(example, attempt_id, agent_type, reflection_memory)` trả về chuỗi câu trả lời từ LLM thật.
2. THE `LLM_Runtime` SHALL cung cấp hàm `evaluator(example, answer)` trả về đối tượng `JudgeResult` được parse từ JSON response của LLM.
3. THE `LLM_Runtime` SHALL cung cấp hàm `reflector(example, attempt_id, judge)` trả về đối tượng `ReflectionEntry` được parse từ JSON response của LLM.
4. THE `LLM_Runtime` SHALL hỗ trợ ít nhất một trong các provider: OpenAI API, Gemini API, hoặc Ollama (local).
5. IF LLM trả về JSON không hợp lệ, THEN THE `LLM_Runtime` SHALL thử parse lại hoặc trả về giá trị mặc định an toàn thay vì để chương trình crash.
6. THE `LLM_Runtime` SHALL trả về `token_count` thực tế từ trường `usage` trong API response (không dùng giá trị ước tính cứng).

---

### Yêu cầu 7: Triển khai Reflexion Loop trong agents.py

**User Story:** Là học viên, tôi muốn triển khai vòng lặp Reflexion trong `BaseAgent.run()`, để `ReflexionAgent` có thể tự cải thiện câu trả lời qua nhiều lần thử.

#### Tiêu chí chấp nhận

1. WHEN `agent_type == "reflexion"` và `judge.score == 0` và `attempt_id < max_attempts`, THE `Reflexion_Agent` SHALL gọi `reflector()` để lấy `ReflectionEntry`.
2. WHEN một `ReflectionEntry` được tạo ra, THE `Reflexion_Agent` SHALL thêm `next_strategy` vào `reflection_memory` để Actor dùng ở lần thử tiếp theo.
3. WHEN một `ReflectionEntry` được tạo ra, THE `Reflexion_Agent` SHALL thêm `ReflectionEntry` vào danh sách `reflections` trong `RunRecord`.
4. WHEN `judge.score == 1` ở bất kỳ lần thử nào, THE `Reflexion_Agent` SHALL dừng vòng lặp ngay lập tức và không gọi thêm `reflector()`.
5. WHEN `agent_type == "react"`, THE `ReAct_Agent` SHALL chỉ thực hiện đúng một lần thử và không gọi `reflector()`.
6. THE `BaseAgent` SHALL tính `token_estimate` của mỗi `AttemptTrace` từ `token_count` thực tế trả về bởi `LLM_Runtime`.
7. THE `BaseAgent` SHALL đo `latency_ms` của mỗi `AttemptTrace` bằng thời gian thực tế của lần gọi LLM tương ứng.

---

### Yêu cầu 8: Chạy Benchmark trên ≥100 mẫu HotpotQA

**User Story:** Là học viên, tôi muốn chạy benchmark với LLM thật trên ít nhất 100 mẫu HotpotQA, để có đủ dữ liệu thực nghiệm so sánh ReAct và Reflexion Agent.

#### Tiêu chí chấp nhận

1. THE `run_benchmark.py` SHALL chạy cả `ReActAgent` và `ReflexionAgent` trên cùng một tập dữ liệu.
2. WHEN chạy benchmark với LLM thật, THE `run_benchmark.py` SHALL truyền `mode="real"` vào hàm `build_report()`.
3. THE benchmark SHALL xử lý ít nhất 100 mẫu dữ liệu từ HotpotQA (tức là `meta.num_records >= 100` trong `report.json`).
4. THE benchmark SHALL ghi lại kết quả của từng mẫu vào file `react_runs.jsonl` và `reflexion_runs.jsonl`.
5. IF một mẫu dữ liệu gây ra lỗi trong quá trình gọi LLM, THEN THE `run_benchmark.py` SHALL ghi log lỗi và tiếp tục xử lý các mẫu còn lại thay vì dừng toàn bộ quá trình.

---

### Yêu cầu 9: Xuất báo cáo đúng định dạng

**User Story:** Là học viên, tôi muốn xuất `report.json` và `report.md` đúng định dạng quy định, để công cụ `autograde.py` có thể chấm điểm tự động.

#### Tiêu chí chấp nhận

1. THE `ReportPayload` SHALL chứa đủ sáu trường bắt buộc: `meta`, `summary`, `failure_modes`, `examples`, `extensions`, `discussion`.
2. THE `meta` SHALL chứa trường `num_records` phản ánh tổng số bản ghi thực tế đã chạy.
3. THE `summary` SHALL chứa kết quả riêng biệt cho `"react"` và `"reflexion"` bao gồm `em`, `avg_attempts`, `avg_token_estimate`, `avg_latency_ms`.
4. THE `failure_modes` SHALL phân loại lỗi theo ít nhất ba loại khác nhau (ví dụ: `none`, `entity_drift`, `incomplete_multi_hop`, `wrong_final_answer`).
5. THE `examples` SHALL chứa ít nhất 20 bản ghi mẫu với đầy đủ các trường theo schema `RunRecord`.
6. THE `discussion` SHALL là chuỗi văn bản có độ dài ít nhất 250 ký tự, phân tích kết quả thực nghiệm.
7. WHEN `save_report()` được gọi, THE `Reflexion_Agent` SHALL ghi `report.json` và `report.md` vào thư mục đầu ra được chỉ định.

---

### Yêu cầu 10: Tính Token thực tế từ API Response

**User Story:** Là học viên, tôi muốn tính số token thực tế từ API response thay vì dùng giá trị ước tính cứng, để báo cáo phản ánh chi phí thực tế của từng agent.

#### Tiêu chí chấp nhận

1. WHEN LLM API trả về trường `usage` trong response, THE `LLM_Runtime` SHALL đọc `usage.total_tokens` (hoặc tương đương) và gán vào `AttemptTrace.token_estimate`.
2. THE `RunRecord.token_estimate` SHALL bằng tổng `token_estimate` của tất cả `AttemptTrace` trong cùng một `RunRecord`.
3. IF API không trả về thông tin `usage`, THEN THE `LLM_Runtime` SHALL dùng thư viện `tiktoken` hoặc phương pháp đếm token tương đương để ước tính, và ghi chú rõ trong code.

---

### Yêu cầu 11 (Bonus): Structured Evaluator

**User Story:** Là học viên, tôi muốn triển khai `structured_evaluator` sử dụng tính năng structured output của LLM, để Evaluator trả về JSON đúng schema mà không cần parse thủ công.

#### Tiêu chí chấp nhận

1. WHERE học viên triển khai `structured_evaluator`, THE `LLM_Runtime` SHALL sử dụng tính năng `response_format` hoặc function calling của LLM API để ép output theo schema `JudgeResult`.
2. WHERE học viên triển khai `structured_evaluator`, THE `ReportPayload.extensions` SHALL chứa chuỗi `"structured_evaluator"`.
3. WHEN `structured_evaluator` được kích hoạt, THE `LLM_Runtime` SHALL không cần bước parse JSON thủ công vì LLM đảm bảo output hợp lệ.

---

### Yêu cầu 12 (Bonus): Reflection Memory nâng cao

**User Story:** Là học viên, tôi muốn triển khai `reflection_memory` nâng cao, để Agent tích lũy và tái sử dụng bài học qua nhiều câu hỏi khác nhau trong cùng một phiên benchmark.

#### Tiêu chí chấp nhận

1. WHERE học viên triển khai `reflection_memory`, THE `Reflexion_Agent` SHALL duy trì một bộ nhớ toàn cục lưu các `lesson` từ các câu hỏi đã xử lý trước đó.
2. WHERE học viên triển khai `reflection_memory`, THE `Reflexion_Agent` SHALL truyền các bài học liên quan vào `reflection_memory` của Actor khi xử lý câu hỏi mới.
3. WHERE học viên triển khai `reflection_memory`, THE `ReportPayload.extensions` SHALL chứa chuỗi `"reflection_memory"`.

---

### Yêu cầu 13 (Bonus): Adaptive Max Attempts

**User Story:** Là học viên, tôi muốn triển khai `adaptive_max_attempts`, để Agent tự điều chỉnh số lần thử tối đa dựa trên độ khó của câu hỏi.

#### Tiêu chí chấp nhận

1. WHERE học viên triển khai `adaptive_max_attempts`, THE `Reflexion_Agent` SHALL đọc trường `difficulty` của `QAExample` để xác định `max_attempts` động (ví dụ: `easy=1`, `medium=2`, `hard=3`).
2. WHERE học viên triển khai `adaptive_max_attempts`, THE `ReportPayload.extensions` SHALL chứa chuỗi `"adaptive_max_attempts"`.

---

### Yêu cầu 14 (Bonus): Memory Compression

**User Story:** Là học viên, tôi muốn triển khai `memory_compression`, để `reflection_memory` không bị quá dài khi số lần thử tăng lên, giúp tiết kiệm token.

#### Tiêu chí chấp nhận

1. WHERE học viên triển khai `memory_compression`, THE `Reflexion_Agent` SHALL tóm tắt `reflection_memory` khi danh sách vượt quá một ngưỡng nhất định (ví dụ: ≥3 mục).
2. WHERE học viên triển khai `memory_compression`, THE `Reflexion_Agent` SHALL dùng LLM hoặc thuật toán đơn giản để nén nhiều bài học thành một đoạn tóm tắt ngắn gọn.
3. WHERE học viên triển khai `memory_compression`, THE `ReportPayload.extensions` SHALL chứa chuỗi `"memory_compression"`.
