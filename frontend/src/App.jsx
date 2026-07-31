import { Fragment, useState } from 'react'

const API_BASE = 'http://localhost:8000'

const MODE_LABELS = {
  locate_extract: 'Locate & Extract',
  summarize_selection: 'Summarize (đoạn chọn)',
  summarize_progress: 'Summarize (tiến độ)',
  clarification: 'Clarification',
}

function ModeBadge({ mode }) {
  return (
    <span className="inline-block text-[10px] font-bold uppercase tracking-wide text-primary bg-primary/10 rounded px-2 py-0.5">
      {MODE_LABELS[mode] || mode}
    </span>
  )
}

// M9: locate_extract/summarize_* trả về "keyword chip" (3-6 cụm ngắn, có
// category) thay vì câu văn/bullet — mapping màu tái dùng token M3 đã có sẵn
// trong index.html theo kiểu "bg-{role}/10 + text-{role}" (giống ModeBadge ở
// trên và citation chip bên dưới), không thêm hex màu mới. "solution" (xanh
// lá) không có token M3 tương ứng (palette gốc không có role xanh lá) nên
// dùng Tailwind green-* thô — có tiền lệ trong mockup (amber-700, red-500
// dùng thô ở vài chỗ ngoài token chính thức).
const CATEGORY_STYLES = {
  problem: 'bg-error/10 text-error border-error/20',
  solution: 'bg-green-600/10 text-green-700 border-green-600/20',
  metric: 'bg-secondary/10 text-secondary border-secondary/20',
  definition: 'bg-tertiary/10 text-tertiary border-tertiary/20',
  none: 'bg-surface-container-high text-on-surface-variant border-outline-variant',
}

function KeywordChips({ keywords }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 max-w-[85%]">
      {keywords.map((kw, idx) => (
        <Fragment key={idx}>
          {idx > 0 && kw.connector && (
            <span className="text-outline text-[13px] font-bold" aria-hidden="true">
              →
            </span>
          )}
          <span
            className={`text-[12px] font-bold px-2.5 py-1 rounded-full border ${CATEGORY_STYLES[kw.category] || CATEGORY_STYLES.none}`}
          >
            {kw.text}
          </span>
        </Fragment>
      ))}
    </div>
  )
}

function ChatMessage({ message, onConfirmClarification, onDismissClarification, onShowDetail }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-primary text-white p-3 rounded-2xl rounded-tr-none font-body-md text-body-md">
          {message.text}
        </div>
      </div>
    )
  }

  const isError = message.mode === 'error'
  const canShowDetail = message.mode === 'locate_extract' && message.citation_page != null
  const hasKeywords = message.keywords && message.keywords.length > 0

  return (
    <div className="flex flex-col items-start gap-2">
      {message.mode && !isError && <ModeBadge mode={message.mode} />}
      {hasKeywords ? (
        <KeywordChips keywords={message.keywords} />
      ) : (
        <div
          className={
            isError
              ? 'max-w-[85%] bg-error-container text-on-error-container p-3 rounded-2xl rounded-tl-none font-body-md text-body-md'
              : 'max-w-[85%] bg-surface-container-high text-on-surface p-3 rounded-2xl rounded-tl-none font-body-md text-body-md'
          }
        >
          {message.text}
        </div>
      )}
      {message.citation_page != null && (
        <div className="flex items-center gap-1.5 text-[11px] font-bold text-primary bg-primary/5 border border-primary/10 rounded-full px-2.5 py-1">
          <span className="material-symbols-outlined text-[14px]">bookmark</span>
          Trích dẫn: trang {message.citation_page}
        </div>
      )}
      {canShowDetail && !message.detail && (
        <button
          type="button"
          onClick={onShowDetail}
          disabled={message.detailLoading}
          className="text-[11px] font-bold text-primary underline decoration-dotted underline-offset-2 hover:no-underline disabled:opacity-40"
        >
          {message.detailLoading ? 'Đang tải chi tiết...' : 'Xem chi tiết hơn'}
        </button>
      )}
      {message.detailError && (
        <p className="text-[11px] text-error">{message.detailError}</p>
      )}
      {message.detail && (
        <div className="max-w-[85%] bg-surface-container text-on-surface-variant p-3 rounded-xl border border-outline-variant font-body-sm text-body-sm">
          {message.detail}
        </div>
      )}
      {message.needsClarification && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onConfirmClarification}
            className="text-[12px] font-bold px-3 py-1.5 border border-primary text-primary rounded-full hover:bg-primary hover:text-white transition-all"
          >
            Giải thích ngay
          </button>
          <button
            type="button"
            onClick={onDismissClarification}
            className="text-[12px] font-bold px-3 py-1.5 border border-outline-variant text-outline rounded-full hover:bg-surface-container-high transition-all"
          >
            Để sau
          </button>
        </div>
      )}
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('idle')
  const [uploadError, setUploadError] = useState(null)
  const [doc, setDoc] = useState(null) // { doc_id, filename, num_pages, num_chunks }
  const [pages, setPages] = useState([])
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedText, setSelectedText] = useState('')
  const [question, setQuestion] = useState('')
  const [useCurrentPageAsHint, setUseCurrentPageAsHint] = useState(false)
  const [messages, setMessages] = useState([]) // { role, mode?, text, citation_page?, needsClarification?, origQuestion? }
  const [chatBusy, setChatBusy] = useState(false)
  const [highlightActive, setHighlightActive] = useState(true)

  const handleUpload = async (e) => {
    e.preventDefault()
    if (!file) return
    setUploadStatus('uploading')
    setUploadError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setDoc(data)
      setUploadStatus('done')

      const pagesRes = await fetch(`${API_BASE}/doc/${data.doc_id}/pages`)
      const pagesData = await pagesRes.json()
      setPages(pagesData.pages)
      setCurrentPage(1)
      setMessages([])
    } catch (err) {
      setUploadError(err.message)
      setUploadStatus('error')
    }
  }

  const activePageText = pages.find((p) => p.page_number === currentPage)?.text || ''

  const handleTextSelect = () => {
    const sel = window.getSelection()?.toString().trim()
    if (sel) setSelectedText(sel)
  }

  const pushMessage = (msg) => setMessages((prev) => [...prev, msg])

  const callChat = async (body, origQuestion) => {
    setChatBusy(true)
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: doc.doc_id, ...body }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Chat failed')
      pushMessage({
        role: 'assistant',
        mode: data.mode,
        text: data.answer,
        keywords: data.keywords,
        citation_page: data.citation_page,
        needsClarification: data.needs_clarification,
        origQuestion,
      })
    } catch (err) {
      pushMessage({ role: 'assistant', mode: 'error', text: `Lỗi: ${err.message}` })
    } finally {
      setChatBusy(false)
    }
  }

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    const q = question.trim()
    pushMessage({ role: 'user', text: q })
    setQuestion('')
    await callChat({ message: q, page_hint: useCurrentPageAsHint ? currentPage : null }, q)
  }

  const handleSummarizeSelection = async () => {
    if (!selectedText) return
    pushMessage({ role: 'user', text: `(Bôi đen) "${selectedText.slice(0, 60)}${selectedText.length > 60 ? '...' : ''}"` })
    await callChat({ selected_text: selectedText })
  }

  const handleSummarizeProgress = async () => {
    pushMessage({ role: 'user', text: `(Tóm tắt tiến độ đến trang ${currentPage})` })
    await callChat({ up_to_page: currentPage })
  }

  // "Giải thích ngay" / "Để sau" map thẳng vào luồng clarification đã có ở
  // backend: gửi lại đúng câu hỏi kèm page_hint (bỏ qua nhánh ambiguous), hoặc
  // không gửi gì và chỉ đóng gợi ý lại.
  const resolveClarification = async (index) => {
    const msg = messages[index]
    if (!msg) return
    setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, needsClarification: false } : m)))
    pushMessage({ role: 'user', text: `Giải thích ngay (dùng trang ${currentPage} làm gợi ý)` })
    await callChat({ message: msg.origQuestion, page_hint: currentPage })
  }

  const dismissClarification = (index) => {
    setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, needsClarification: false } : m)))
    pushMessage({ role: 'user', text: 'Để sau' })
  }

  const showDetail = async (index) => {
    const msg = messages[index]
    if (!msg || msg.citation_page == null) return
    setMessages((prev) => prev.map((m, i) => (i === index ? { ...m, detailLoading: true, detailError: null } : m)))
    try {
      const res = await fetch(`${API_BASE}/chat/detail`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: doc.doc_id, page: msg.citation_page }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Không lấy được chi tiết')
      setMessages((prev) =>
        prev.map((m, i) => (i === index ? { ...m, detail: data.detail, detailLoading: false } : m))
      )
    } catch (err) {
      setMessages((prev) =>
        prev.map((m, i) => (i === index ? { ...m, detailLoading: false, detailError: `Lỗi: ${err.message}` } : m))
      )
    }
  }

  if (!doc) {
    return (
      <div className="h-screen w-screen flex flex-col bg-background text-on-background overflow-hidden">
        <header className="flex items-center px-gutter h-16 bg-surface border-b border-outline-variant shrink-0">
          <h1 className="font-headline-md text-headline-md font-extrabold text-primary">AI Study Companion</h1>
        </header>
        <main className="flex-1 flex items-center justify-center p-6">
          <form
            onSubmit={handleUpload}
            className="w-full max-w-md bg-surface-container-lowest border border-outline-variant rounded-xl shadow-sm p-8 flex flex-col gap-4"
          >
            <div className="flex flex-col gap-1">
              <h2 className="font-headline-sm text-headline-sm text-primary">Tải lên bài giảng PDF</h2>
              <p className="font-body-sm text-body-sm text-outline">
                Upload 1 file PDF/slide để bắt đầu theo dõi cùng AI Tutor.
              </p>
            </div>
            <label className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-outline-variant rounded-lg py-8 px-4 cursor-pointer hover:border-primary hover:bg-primary/5 transition-colors">
              <span className="material-symbols-outlined text-primary text-[32px]">upload_file</span>
              <span className="font-label-bold text-label-bold text-on-surface-variant text-center">
                {file ? file.name : 'Chọn file PDF'}
              </span>
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button
              type="submit"
              disabled={!file || uploadStatus === 'uploading'}
              className="w-full bg-primary text-white py-3 rounded-lg font-label-bold text-label-bold hover:bg-primary-container transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {uploadStatus === 'uploading' ? 'Đang tải lên...' : 'Upload PDF'}
            </button>
            {uploadError && <p className="text-body-sm font-body-sm text-error">Lỗi: {uploadError}</p>}
          </form>
        </main>
      </div>
    )
  }

  return (
    <div className="h-screen w-screen flex flex-col bg-background text-on-background overflow-hidden">
      <header className="flex justify-between items-center w-full px-gutter h-16 z-50 bg-surface border-b border-outline-variant shrink-0">
        <h1 className="font-headline-md text-headline-md font-extrabold text-primary">AI Study Companion</h1>
        <div className="flex items-center gap-2 font-body-sm text-body-sm text-outline">
          <span className="material-symbols-outlined text-[18px]">description</span>
          <span>
            {doc.filename} · {doc.num_pages} trang
          </span>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden relative">
        {/* Left: page navigation */}
        <aside className="fixed left-0 top-16 bottom-0 w-sidebar-width flex flex-col z-40 bg-surface-container border-r border-outline-variant">
          <div className="p-6">
            <h2 className="font-headline-sm text-headline-sm text-primary mb-1 truncate" title={doc.filename}>
              {doc.filename}
            </h2>
            <p className="font-body-sm text-body-sm text-outline">
              {doc.num_pages} trang · {doc.num_chunks} đoạn đã lập chỉ mục
            </p>
          </div>
          <nav className="flex-1 overflow-y-auto px-3 py-2 space-y-1 custom-scrollbar">
            <div className="px-3 mb-2">
              <span className="text-[10px] uppercase tracking-wider font-bold text-outline">Danh sách trang</span>
            </div>
            {pages.map((p) => {
              const active = p.page_number === currentPage
              return (
                <button
                  key={p.page_number}
                  type="button"
                  onClick={() => setCurrentPage(p.page_number)}
                  className={
                    active
                      ? 'w-full flex items-center gap-3 bg-primary-container text-on-primary-container font-bold rounded-lg px-4 py-3 active-tab-shadow translate-x-1 transition-all duration-200'
                      : 'w-full flex items-center gap-3 text-on-surface-variant px-4 py-3 hover:bg-surface-container-high rounded-lg transition-all group'
                  }
                >
                  <span
                    className="material-symbols-outlined text-[20px]"
                    style={active ? { fontVariationSettings: "'FILL' 1" } : undefined}
                  >
                    {active ? 'auto_awesome' : 'description'}
                  </span>
                  <span className="font-label-bold text-label-bold truncate">Trang {p.page_number}</span>
                </button>
              )
            })}
          </nav>
        </aside>

        {/* Center: PDF viewer */}
        <section className="flex-1 ml-sidebar-width mr-chat-width bg-surface-container-low flex flex-col relative overflow-hidden">
          <div className="h-14 bg-surface border-b border-outline-variant flex items-center justify-between px-6 z-30 shrink-0 overflow-x-auto">
            <div className="flex items-center gap-1">
              <button type="button" className="p-2 text-primary bg-primary-container/20 rounded-lg flex items-center gap-2 px-3">
                <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: "'FILL' 1" }}>
                  visibility
                </span>
                <span className="font-label-bold text-label-bold">Đọc</span>
              </button>
              <button
                type="button"
                className="p-2 hover:bg-surface-container rounded-lg text-on-surface-variant flex items-center gap-2 px-3"
              >
                <span className="material-symbols-outlined text-[20px]">edit</span>
                <span className="font-label-bold text-label-bold">Bút</span>
              </button>
              <button
                type="button"
                onClick={() => setHighlightActive((v) => !v)}
                title="Bật/tắt bôi đen để hỏi AI"
                className={
                  highlightActive
                    ? 'p-2 text-primary bg-primary-container/20 rounded-lg flex items-center gap-2 px-3'
                    : 'p-2 hover:bg-surface-container rounded-lg text-on-surface-variant flex items-center gap-2 px-3'
                }
              >
                <span
                  className="material-symbols-outlined text-[20px]"
                  style={highlightActive ? { fontVariationSettings: "'FILL' 1" } : undefined}
                >
                  highlighter_size_3
                </span>
                <span className="font-label-bold text-label-bold">Highlight</span>
              </button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center bg-surface-container-high rounded-lg px-2 py-1">
                <button type="button" className="p-1 hover:bg-surface-container-highest rounded transition-colors">
                  <span className="material-symbols-outlined text-[18px]">remove</span>
                </button>
                <span className="px-3 font-label-bold text-label-bold">100%</span>
                <button type="button" className="p-1 hover:bg-surface-container-highest rounded transition-colors">
                  <span className="material-symbols-outlined text-[18px]">add</span>
                </button>
              </div>
              <div className="flex items-center gap-2 font-label-bold text-label-bold text-on-surface-variant">
                <span className="bg-surface-container-high px-2 py-1 rounded">{currentPage}</span>
                <span>/ {doc.num_pages}</span>
              </div>
              <div className="h-6 w-[1px] bg-outline-variant" />
              <button
                type="button"
                onClick={handleSummarizeProgress}
                disabled={chatBusy}
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-on-surface-variant hover:bg-surface-container transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <span className="material-symbols-outlined text-[20px]">summarize</span>
                <span className="font-label-bold text-label-bold">Tóm tắt đến trang {currentPage}</span>
              </button>
              <div className="h-6 w-[1px] bg-outline-variant" />
              <button type="button" className="p-2 hover:bg-surface-container rounded-lg text-on-surface-variant">
                <span className="material-symbols-outlined text-[20px]">download</span>
              </button>
              <button type="button" className="p-2 hover:bg-surface-container rounded-lg text-on-surface-variant">
                <span className="material-symbols-outlined text-[20px]">print</span>
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-auto p-8 custom-scrollbar flex justify-center bg-surface-dim/30">
            <div className="w-full max-w-[900px] min-h-[600px] bg-white shadow-xl rounded-sm relative flex flex-col">
              <div className="absolute top-0 left-0 w-full h-2 bg-primary" />
              <div
                onMouseUp={highlightActive ? handleTextSelect : undefined}
                className="p-12 flex-1 whitespace-pre-wrap font-body-lg text-body-lg text-on-surface"
              >
                {activePageText || <em className="text-outline">(trang trống)</em>}
              </div>
              <div className="p-6 border-t border-surface-container-high flex justify-between items-center text-[11px] text-outline font-medium">
                <span>{doc.filename}</span>
                <span>
                  Trang {currentPage} / {doc.num_pages}
                </span>
              </div>
            </div>
          </div>

          {selectedText && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 bg-surface-container-lowest border border-outline-variant shadow-lg rounded-xl px-4 py-3 flex items-center gap-4 max-w-[min(90%,640px)]">
              <span className="material-symbols-outlined text-primary text-[20px] shrink-0" style={{ fontVariationSettings: "'FILL' 1" }}>
                highlighter_size_3
              </span>
              <p className="text-body-sm font-body-sm text-on-surface-variant truncate">
                Đang chọn: <em>"{selectedText.slice(0, 80)}{selectedText.length > 80 ? '...' : ''}"</em>
              </p>
              <button
                type="button"
                onClick={handleSummarizeSelection}
                disabled={chatBusy}
                className="shrink-0 text-[12px] font-bold px-3 py-1.5 bg-primary text-white rounded-full hover:bg-primary-container transition-all disabled:opacity-40"
              >
                Tóm tắt đoạn này
              </button>
            </div>
          )}
        </section>

        {/* Right: AI chat */}
        <aside className="fixed right-0 top-16 bottom-0 w-chat-width flex flex-col z-40 glass-panel border-l border-outline-variant shadow-sm">
          <div className="p-4 border-b border-outline-variant flex items-center justify-between shrink-0">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-on-primary-container flex items-center justify-center shadow-md">
                <span className="material-symbols-outlined text-white text-[22px]" style={{ fontVariationSettings: "'FILL' 1" }}>
                  smart_toy
                </span>
              </div>
              <div>
                <h3 className="font-headline-sm text-[16px] text-primary leading-none">AI Tutor</h3>
                <p className="text-[10px] text-on-tertiary-container font-semibold uppercase mt-1">Trợ lý học tập</p>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
            {messages.length === 0 && (
              <p className="text-body-sm font-body-sm text-outline text-center mt-8">
                Chưa có tin nhắn nào. Hỏi trực tiếp hoặc bôi đen tài liệu để bắt đầu.
              </p>
            )}
            {messages.map((m, i) => (
              <ChatMessage
                key={i}
                message={m}
                onConfirmClarification={() => resolveClarification(i)}
                onDismissClarification={() => dismissClarification(i)}
                onShowDetail={() => showDetail(i)}
              />
            ))}
          </div>

          <div className="p-4 bg-white border-t border-outline-variant shrink-0">
            <form onSubmit={handleAsk} className="relative group">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Nhập câu hỏi hoặc bôi đen tài liệu..."
                rows={2}
                className="w-full bg-surface-container-low border-outline-variant rounded-xl p-4 pr-12 focus:ring-2 focus:ring-on-primary-container focus:border-transparent text-body-md resize-none transition-all group-focus-within:bg-white"
              />
              <button
                type="submit"
                disabled={chatBusy || !question.trim()}
                className="absolute right-3 bottom-3 p-2 bg-primary text-white rounded-lg hover:bg-primary-container transition-all active:scale-90 flex items-center justify-center disabled:opacity-40"
              >
                <span className="material-symbols-outlined text-[20px]">send</span>
              </button>
            </form>
            <label className="flex items-center gap-2 mt-3 px-1 text-[11px] font-medium text-outline">
              <input
                type="checkbox"
                checked={useCurrentPageAsHint}
                onChange={(e) => setUseCurrentPageAsHint(e.target.checked)}
                className="rounded border-outline-variant text-primary focus:ring-primary"
              />
              Dùng trang {currentPage} đang xem làm gợi ý (page_hint)
            </label>
          </div>
        </aside>
      </main>
    </div>
  )
}

export default App
