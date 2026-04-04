document.addEventListener('DOMContentLoaded', () => {
    const pdfInput = document.getElementById('pdfInput');
    const dropZone = document.getElementById('dropZone');
    const processBtn = document.getElementById('processBtn');
    const output = document.getElementById('output');
    const loading = document.getElementById('loading');
    const stepLabel = document.getElementById('stepLabel');

    const BUCKET_ORDER = ['Objective', 'Methodology', 'Results', 'Conclusion'];
    const POLL_INTERVAL_MS = 3000;

    let selectedFile = null;
    let isProcessing = false;

    if (!pdfInput || !dropZone || !processBtn || !output || !loading || !stepLabel) {
        console.error('Critical: required DOM elements missing.');
        return;
    }

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

    processBtn.addEventListener('click', async () => {
        if (isProcessing) {
            return;
        }

        const file = selectedFile || pdfInput.files[0];
        if (!file) {
            alert('Please select or drop a PDF first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        isProcessing = true;
        loading.classList.remove('hidden');
        output.classList.add('hidden');
        output.innerHTML = '';
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        dropZone.style.opacity = '0.65';
        dropZone.style.pointerEvents = 'none';
        stepLabel.textContent = 'Uploading PDF and creating job...';

        try {
            const submitResponse = await fetch('/api/process-paper', {
                method: 'POST',
                body: formData,
            });

            const submitData = await parseJsonResponse(submitResponse, 'Server returned a non-JSON response while creating the job.');
            if (!submitResponse.ok) {
                throw new Error(submitData?.detail || 'Paper processing failed to start.');
            }

            const result = await pollJobUntilComplete(submitData.job_id);
            renderReport(result);
        } catch (err) {
            console.error('Upload Error:', err);
            stepLabel.textContent = 'Processing failed.';
            alert(`Error: ${err.message}`);
        } finally {
            isProcessing = false;
            loading.classList.add('hidden');
            processBtn.disabled = false;
            processBtn.textContent = 'Process Paper';
            dropZone.style.opacity = '';
            dropZone.style.pointerEvents = '';
        }
    });

    async function pollJobUntilComplete(jobId) {
        if (!jobId) {
            throw new Error('Missing job ID from server.');
        }

        while (true) {
            const response = await fetch(`/api/process-paper/${encodeURIComponent(jobId)}`);
            const data = await parseJsonResponse(response, 'Server returned a non-JSON response while checking job status.');

            if (!response.ok) {
                throw new Error(data?.detail || 'Failed to fetch job status.');
            }

            stepLabel.textContent = data?.step || 'Processing...';

            if (data.status === 'completed') {
                stepLabel.textContent = 'Completed.';
                return data;
            }

            if (data.status === 'failed') {
                throw new Error(data?.error || 'Paper processing failed.');
            }

            await delay(POLL_INTERVAL_MS);
        }
    }

    function renderReport(data) {
        const paperTitle = data?.paper_title || 'Untitled Paper';
        const summary = data?.summary || {};

        let html = `<h1 class="paper-main-title">${escapeHtml(paperTitle)}</h1>`;

        if (Object.keys(summary).length === 0) {
            html += '<div class="bucket-section"><p class="empty-note">No summary was generated.</p></div>';
        }

        BUCKET_ORDER.forEach((bucket) => {
            const bucketData = summary[bucket];
            const points = Array.isArray(bucketData?.points) ? bucketData.points : [];
            const rawText = typeof bucketData?.raw_text === 'string' ? bucketData.raw_text : '';
            const images = Array.isArray(bucketData?.images) ? bucketData.images : [];

            html += `<div class="bucket-section">`;
            html += `<h2>${escapeHtml(bucket)}</h2>`;

            if (points.length > 0) {
                html += `<ol class="points-list">`;
                points.forEach((point, i) => {
                    html += `
                        <li>
                            <span class="point-num">${i + 1}</span>
                            <span>${escapeHtml(stripInlineFormatting(point))}</span>
                        </li>`;
                });
                html += `</ol>`;
            } else {
                html += `<p class="empty-note">No content generated for this section.</p>`;
            }

            if (rawText.trim().length > 0) {
                const toggleId = `raw-${bucket.toLowerCase()}`;
                html += `
                    <button class="toggle-raw-btn" onclick="toggleRaw('${toggleId}', this)">
                        <span>Show extracted sentences</span>
                        <span class="arrow">▼</span>
                    </button>
                    <div id="${toggleId}" class="raw-text-panel">${escapeHtml(stripInlineFormatting(rawText))}</div>`;
            }

            if (images.length > 0) {
                html += `<div class="figure-grid">`;
                images.forEach((img) => {
                    const src = normalizePath(img?.path || '');
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

    async function parseJsonResponse(response, errorMessage) {
        try {
            return await response.json();
        } catch (_) {
            throw new Error(errorMessage);
        }
    }

    function delay(ms) {
        return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    function normalizePath(path) {
        if (!path) return '';
        if (/^https?:\/\//i.test(path)) return path;
        return path.startsWith('/') ? path : `/${path}`;
    }

    function stripInlineFormatting(text) {
        return String(text).replace(/<\/?strong>/gi, '');
    }

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

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
