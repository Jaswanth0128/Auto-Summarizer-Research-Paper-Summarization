document.addEventListener('DOMContentLoaded', () => {
    const pdfInput  = document.getElementById('pdfInput');
    const dropZone  = document.getElementById('dropZone');
    const processBtn = document.getElementById('processBtn');
    const output    = document.getElementById('output');
    const loading   = document.getElementById('loading');
    const stepLabel = document.getElementById('stepLabel');

    const BUCKET_ORDER = ['Objective', 'Methodology', 'Results', 'Conclusion'];

    // Step labels shown during processing
    const STEPS = [
        'Extracting text and tables from PDF…',
        'Classifying sections into buckets…',
        'Cleaning and deduplicating sentences…',
        'Phi-3 Mini is rewriting — this may take 30–60 seconds…',
    ];

    let selectedFile = null;
    let stepInterval = null;

    if (!pdfInput || !dropZone || !processBtn || !output || !loading) {
        console.error('Critical: required DOM elements missing.');
        return;
    }

    // ── File selection ──────────────────────────────────────
    dropZone.addEventListener('click', () => pdfInput.click());

    pdfInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            handleFileSelection(selectedFile);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '#eff6ff';
        dropZone.style.borderColor = '#2563eb';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.backgroundColor = '';
        dropZone.style.borderColor = '';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.backgroundColor = '';
        dropZone.style.borderColor = '';

        const files = e.dataTransfer.files;
        if (!files || files.length === 0) return;

        const dropped = files[0];
        if (dropped.type !== 'application/pdf' && !dropped.name.toLowerCase().endsWith('.pdf')) {
            alert('Please drop a valid PDF file.');
            return;
        }

        selectedFile = dropped;
        try { pdfInput.files = files; } catch (_) {}
        handleFileSelection(selectedFile);
    });

    function handleFileSelection(file) {
        dropZone.innerHTML = `<p><strong>Ready:</strong> ${escapeHtml(file.name)}</p>`;
        processBtn.disabled = false;
    }

    // ── Submit ───────────────────────────────────────────────
    processBtn.addEventListener('click', async () => {
        const file = selectedFile || pdfInput.files[0];
        if (!file) {
            alert('Please select or drop a PDF first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        loading.classList.remove('hidden');
        output.classList.add('hidden');
        processBtn.disabled = true;

        // Cycle through step labels so user knows something is happening
        let stepIndex = 0;
        if (stepLabel) stepLabel.textContent = STEPS[0];
        stepInterval = setInterval(() => {
            stepIndex = Math.min(stepIndex + 1, STEPS.length - 1);
            if (stepLabel) stepLabel.textContent = STEPS[stepIndex];
        }, 12000);  // advance label every 12 seconds

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout
            const response = await fetch("/api/process-paper", {
              method: "POST",
              body: formData,
              signal: controller.signal,
            });
            clearTimeout(timeoutId);

            let data = null;
            try { data = await response.json(); }
            catch (_) { throw new Error('Server returned a non-JSON response.'); }

            if (!response.ok) {
                throw new Error(data?.detail || 'Paper processing failed.');
            }

            renderReport(data);

        } catch (err) {
            console.error('Upload Error:', err);
            alert(`Error: ${err.message}`);
        } finally {
            clearInterval(stepInterval);
            loading.classList.add('hidden');
            processBtn.disabled = false;
        }
    });

    // ── Render ───────────────────────────────────────────────
    function renderReport(data) {
        const paperTitle = data?.paper_title || 'Untitled Paper';
        const summary    = data?.summary || {};

        let html = `<h1 class="paper-main-title">${escapeHtml(paperTitle)}</h1>`;

        if (Object.keys(summary).length === 0) {
            html += '<div class="bucket-section"><p class="empty-note">No summary was generated.</p></div>';
        }

        BUCKET_ORDER.forEach((bucket) => {
            const bucketData = summary[bucket];
            const points  = Array.isArray(bucketData?.points)  ? bucketData.points  : [];
            const rawText = typeof bucketData?.raw_text === 'string' ? bucketData.raw_text : '';
            const images  = Array.isArray(bucketData?.images)  ? bucketData.images  : [];

            html += `<div class="bucket-section">`;
            html += `<h2>${escapeHtml(bucket)}</h2>`;

            // Numbered points
            if (points.length > 0) {
                html += `<ol class="points-list">`;
                points.forEach((point, i) => {
                    html += `
                        <li>
                            <span class="point-num">${i + 1}</span>
                            <span>${escapeHtml(highlightNumbers(point))}</span>
                        </li>`;
                });
                html += `</ol>`;
            } else {
                html += `<p class="empty-note">No content generated for this section.</p>`;
            }

            // Toggle button for raw extracted text
            if (rawText.trim().length > 0) {
                const toggleId = `raw-${bucket.toLowerCase()}`;
                html += `
                    <button class="toggle-raw-btn" onclick="toggleRaw('${toggleId}', this)">
                        <span>Show extracted sentences</span>
                        <span class="arrow">▼</span>
                    </button>
                    <div id="${toggleId}" class="raw-text-panel">${escapeHtml(rawText)}</div>`;
            }

            // Images
            if (images.length > 0) {
                html += `<div class="figure-grid">`;
                images.forEach((img) => {
                    const src     = normalizePath(img?.path || '');
                    const caption = img?.caption || 'Figure/Table';
                    if (src) {
                        html += `
                            <div class="figure-card">
                                <img src="${src}" alt="${escapeHtml(caption)}" loading="lazy">
                                <p class="caption"><em>${escapeHtml(caption)}</em></p>
                            </div>`;
                    }
                });
                html += `</div>`;
            }

            html += `</div>`;
        });

        output.innerHTML = html;
        output.classList.remove('hidden');
        output.scrollIntoView({ behavior: 'smooth' });
    }

    // ── Helpers ──────────────────────────────────────────────
    function normalizePath(path) {
        if (!path) return '';
        if (/^https?:\/\//i.test(path)) return path;
        return path.startsWith('/') ? path : `/${path}`;
    }

    // Wrap numbers/percentages in <strong> for emphasis
    function highlightNumbers(text) {
        return text.replace(/(\b\d+(?:\.\d+)?%?\b)/g, '<strong>$1</strong>');
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g,  '&amp;')
            .replace(/</g,  '&lt;')
            .replace(/>/g,  '&gt;')
            .replace(/"/g,  '&quot;')
            .replace(/'/g,  '&#39;');
    }

    // Exposed globally so inline onclick can reach it
    window.toggleRaw = function(panelId, btn) {
        const panel = document.getElementById(panelId);
        if (!panel) return;
        const isOpen = panel.classList.toggle('visible');
        btn.classList.toggle('open', isOpen);
        btn.querySelector('span:first-child').textContent = isOpen
            ? 'Hide extracted sentences'
            : 'Show extracted sentences';
    };
});
